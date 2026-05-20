import pandas as pd

final_schema = pd.DataFrame(columns = ['category', 'source', 'forbidden_prompt'])

def reader(file_source: str, file):
    try:
        if file_source == 'SGBench':
            sg_bench = pd.read_json(file)
            
            
            final_schema['category'] = sg_bench['safety_type'].str.extract(r'\[([^\]]+)\]').iloc[:, 0].str.split(pat = ":", n = 1).apply(lambda x: x[1])
            final_schema['source'] = 'SGBench'
            final_schema['forbidden_prompt'] = sg_bench['query']
            
        elif file_source == 'StrongREJECT':
            strong_reject = pd.read_csv(file)
            strong_reject['source'] = 'strongREJECT'
            
            final_schema = strong_reject.copy()
            
        return final_schema
        
    except Exception as e:
        raise e