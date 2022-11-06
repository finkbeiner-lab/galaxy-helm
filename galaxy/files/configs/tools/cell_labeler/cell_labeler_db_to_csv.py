import sys
import os
import sqlite3 as lite
import csv
import argparse

TABLE_NAME = 'BioMedImages'

def db_to_csv(input_db, output_csv):
    # Output query result to csv file from database    
    # "ValidFromStartTimePoint = 1" Because we only label and care about the objects originate from start(not necessarily T0, maybe T1)    
    conn = lite.connect(input_db)
    with conn:
        cur = conn.cursor()
        # substr(WellID,1, 1), abs(substr(WellID, 2)) to get better order of WellID. Since simple order by WellID will make A12 comes before A2
        cur.execute("SELECT * FROM {} WHERE ValidFromStartTimePoint = 1 AND Phenotype IS NOT NULL ORDER BY ExperimentName, substr(WellID,1, 1), abs(substr(WellID, 2)), ObjectID, Channel, NumOfHours, ZIndex, BurstIndex".format(TABLE_NAME))
        # Get a list of column names in sqlite
        col_names = [description[0] for description in cur.description]
        rows = cur.fetchall()
        # with open(os.path.join(OUTPUT_CSV_DIR, where_clause+'_'+user_name+'.csv'), 'wb') as csvfile:
        # Make sure Galaxy not show error
        # If you write csv only the header without content, Galaxy will complain in history pane. So you have to write nothing then Galaxy will be satisfied with that and just say "empty" in the history pane
        if rows: 
            with open(output_csv, 'wb') as csvfile: 
                # It's delimiter and not delimeter
                writer = csv.writer(csvfile, delimiter=',')
                # Write header
                writer.writerow(col_names)
                for row in rows:
                    writer.writerow(row)        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_db')
    parser.add_argument('output_csv')
    parser.add_argument('out_report')
    args = parser.parse_args()

    # Get args from argparse
    input_db = args.input_db
    output_csv = args.output_csv
    out_report = args.out_report

    # Check if db file exist
    if not os.path.isfile(input_db):
        print input_db + ' does NOT exist!'
        sys.exit(1)

    # Output db to csv
    db_to_csv(input_db, output_csv)

    # Write output report to Galaxy
    with open(out_report, 'wb') as f:
        f.write('Input database path: %s\n' % input_db)
        f.write('Output csv path: %s\n' % output_csv)
        f.write('CSV notation Notes:')
        f.write('Column Phenotype 0 means neuron')
        f.write('Column Phenotype 1 means glia')
        f.write('Column Phenotype 2 means debris')
        f.write('Column Live 0 means dead')
        f.write('Column Live 1 means live')
        f.write('Column Live 2 means uncertain')










