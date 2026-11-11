"""Pin official per-detector effective-area tables and record their provenance."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parent
REPOSITORY="RomanSpaceTelescope/roman-technical-information"
DATA_PATH="roman_technical_information/data/WideFieldInstrument/Imaging/EffectiveAreas"
DESTINATION=ROOT/"external_reference_data/roman_effarea_v8_20240301"


def fetch(url):
    with urlopen(Request(url,headers={"User-Agent":"CAR201-ice-modelling"}),timeout=60) as response:
        return response.read()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector",action="append",required=True,choices=[f"WFI{i:02d}" for i in range(1,19)])
    parser.add_argument("--revision",help="Pinned repository commit; defaults to resolving main once")
    args=parser.parse_args()
    revision=args.revision
    if revision is None:
        revision=json.loads(fetch(f"https://api.github.com/repos/{REPOSITORY}/commits/main"))["sha"]
    DESTINATION.mkdir(parents=True,exist_ok=True)
    records=[]
    for detector in args.detector:
        name=f"Roman_effarea_v8_SCA{int(detector[3:]):02d}_20240301.ecsv"
        url=f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{DATA_PATH}/{name}"
        content=fetch(url)
        if not content.startswith(b"# %ECSV"):raise ValueError("Not an ECSV effective-area table")
        destination=DESTINATION/name
        destination.write_bytes(content)
        record={"detector":detector,"repository":REPOSITORY,"commit":revision,"url":url,
                "sha256":hashlib.sha256(content).hexdigest(),"size_bytes":len(content),
                "retrieved_utc":datetime.now(timezone.utc).isoformat(),
                "wavelength_unit":"micron","effective_area_unit":"m^2"}
        destination.with_suffix(".provenance.json").write_text(json.dumps(record,indent=2)+"\n")
        records.append(record)
        print(destination,record["sha256"])
    print("Pinned repository commit:",revision)


if __name__=="__main__":
    main()
