from array_record.python import array_record_module  #pyrefly:ignore
import pickle
import grain
from tqdm import tqdm
import os


base_dir = "data/common"

invalid_records = dict()
for i in range(5573):
    invalid_records[str(i)] = False

total_records: int = 14575512
records_per_shard: int = 3643878
current_record_count: int = 0
shard_number:int = 0
write_path = f"data/common_canvas_{shard_number}.array_record"
writer = array_record_module.ArrayRecordWriter(write_path, "group_size:1")

for path in tqdm(os.listdir(base_dir)):
    new_path = os.path.join(base_dir, path)
    num = path.split("_")[-2].split(".")[0]
    try:
        array_record_data_source = grain.sources.ArrayRecordDataSource(new_path)
        for data in array_record_data_source:
            element = pickle.loads(data)
            writer.write(pickle.dumps(element))
            current_record_count += 1
            if current_record_count >= records_per_shard:
                writer.close()
                shard_number += 1                
                current_record_count = 0
                write_path = f"data/common_canvas_{shard_number}.array_record"
                writer = array_record_module.ArrayRecordWriter(write_path, "group_size:1")
    except:
        invalid_records[num] = True

for key, value in invalid_records.items():
    if value:
        print(key, end=" ")
print()
writer.close()
