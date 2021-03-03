import datetime
import psycopg2
import pandas as pd
import json
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures.process import ProcessPoolExecutor

# conn = psycopg2.connect(dbname='siris', user='postgres',
#                         host='testbed.inode.igd.fraunhofer.de', password='vdS83DJSQz2xQ', port=18001)
# tables=["countries",
# "institutions",
# "activity_types",
# "ec_framework_programs",
# "erc_panels",
# "erc_research_domains",
# "eu_territorial_units",
# "funding_schemes",
# "people",
# "programmes",
# "project_erc_panels",
# "project_member_roles",
# "project_members",
# "project_neighbours",
# "project_programmes",
# "projects",
# "project_subject_areas",
# "project_topics",
# "subject_areas",
# "topics",
# "vat_numbers"]

conn = psycopg2.connect(dbname='skyserver_dr16_2020_11_30', user='postgres',
                        host='testbed.inode.igd.fraunhofer.de', password='vdS83DJSQz2xQ', port=18001)
tables = ["galspecline",
"neighbors",
"photoobj",
"specobj",
"spplines"]

def get_value(index, column, table, bin_size, conn):
    value = pd.read_sql(
                f"select {column} from {table} where {column} is not null order by {column} offset { index*bin_size  } limit 1;", conn).iloc[0][column]
    print(value, end='   ')
    return value


# columns = ["rowc_u", "g", "r", "i", "z", "petrorad_r"]
bin_count = 10 


result = {}
for table in tables:
    columns = list(pd.read_sql(f"SELECT column_name  FROM INFORMATION_SCHEMA. COLUMNS WHERE TABLE_NAME = '{table}' and data_type in ('date', 'smallint', 'integer', 'decimal', 'numeric', 'real', 'double precision', 'smallserial', 'serial', 'bigserial', 'bigint', 'real');", conn)["column_name"])
    for column in columns:
        if not "_id" in column:
            item_count = pd.read_sql(
                f"select count(*) from {table} where {column} is not null;", conn).iloc[0]["count"]
            bin_size = (item_count // bin_count) -1
            distinct_values = pd.read_sql(
                f"select count(distinct {column}) from {table} ;", conn).iloc[0]["count"]
            null_count =  pd.read_sql(
                f"select count(*)  from {table} where {column} is null", conn).iloc[0]["count"]
            print(f"Column: {column}, {distinct_values} distinct val, {null_count} null values")
            # if null_count == 0:
            bins = []
            if distinct_values > 50:  
                futures = []
                values = []
                with ThreadPoolExecutor(max_workers=11) as executor:
                    for i in range(11):
                        futures.append(executor.submit(get_value, i, column, table, bin_size, conn))

                    for future in futures:
                        values.append(future.result())

                    values.sort()
                    if type(values[0]) == datetime.date:
                        values = map(lambda x: x.isoformat(), values)
                    else:
                        values = map(lambda x: float(x), values)
                    for index, value in enumerate(values):
                        if index > 0:
                            bins.append((last_value, value))   
                        last_value = value
            else:
                bins = list(pd.read_sql(
                    f"select distinct {column} from {table} where {column} is not null order by {column};", conn)[column] )
                if type(bins[0]) == datetime:
                        bins = map(lambda x: x.isoformat(), bins)
            
            result[f"{table}.{column}"] = bins
            print(bins)
                    #  tt = pd.read_sql(
                    #     f"select min(bin.u), max(bin.u) from (select u from photoobj order by u offset { i*bin_size } limit {bin_size}) as bin;", conn)
            # for i in range(11):
            #     value = pd.read_sql(
            #         f"select {column} from photoobj order by {column} offset { i*bin_size } limit 1;", conn).iloc[0][column]
            #     print(value)
            #     if i > 0:
            #         bins.append((last_value, value))    
            #     last_value = value    
    
with open('bins.json', 'w') as fp:
    json.dump(result, fp, indent=1)    
