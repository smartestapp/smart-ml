# For testing/verification purpose, run the following command
# a prediction.xlsx file will be generated

for kit in ACON_Ab DeepBlue_Ag
do
    python main.py --kit-id=${kit}
done
