import json
import pandas as pd
d= None
with open('bins.json') as f:
    d = json.load(f)
    print(d)
intervals = []
for i in d["photoobj.u"]:
    intervals.append(pd.Interval(i[0], i[1]))
    print(intervals)