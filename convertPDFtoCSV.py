#pip install tabula-py
#pip install pandas
import tabula
import pandas as pd
import asyncio
#tabula.convert_into("2012-1"+".pdf", "2012-1"+".csv", output_format="csv", pages='all', stream=True)
#print("2012-1 "+"converted")

from pathlib import Path

for p in Path('.').glob('*.pdf'):
    print(f"{p.name}\n")
    try:
        tabula.convert_into(str(p.name), str(p.name)[:-4]+".csv", output_format="csv", pages='all', stream=True)
    except:
        print(f"{p.name}"+" NOT CONVERTED"+"\n")