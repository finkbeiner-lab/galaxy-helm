
import sys
import os
import sqlite3 as lite
import csv
import argparse



# DB_NAME = '/Users/guangzhili/GladStone/bio_images_all.db'
# DB_NAME = '/Users/guangzhili/GladStone/galaxy-dev/tools/database_tools/BioImagesDB/bio_images_all.db'  
# DB_NAME = '/Users/guangzhili/GladStone/bio_images_t.db' 
DB_NAME = '/media/robodata/Guangzhi/ImagesDB/bio_images_all.db'   
# TABLE_NAME = 'Images'





def query_db_to_csv(db_arg, table_arg, user_name, experiment_names, well_ids, obj_ids, channels, phenotypes, lives, output_csv):
	''' output query result to csv file based on user query input  '''

	where_clause = []

	if experiment_names:
		if len(experiment_names) == 1:
			where_clause.append("ExperimentName = '%s'" % experiment_names[0])
		else:
			where_clause.append("ExperimentName IN %s" % (tuple(experiment_names),))	
	if well_ids:
		if len(well_ids) == 1:
			where_clause.append("WellID = '%s'" % well_ids[0])
		else:
			where_clause.append("WellID IN %s" % (tuple(well_ids),))	
	if obj_ids:
		if len(obj_ids) == 1:
			where_clause.append("ObjectID = %d" % obj_ids[0])
		else:
			where_clause.append("ObjectID IN %s" % (tuple(obj_ids),))	
	if channels:
		if len(channels) == 1:
			where_clause.append("Channel = '%s'" % channels[0])
		else:
			where_clause.append("Channel IN %s" % (tuple(channels),))	
	if phenotypes: # Be careful of the difference between 0 and '0' at command line argument
		if len(phenotypes) == 1:
			where_clause.append("Phenotype = %d" % phenotypes[0])
		else:
			where_clause.append("Phenotype IN %s" % (tuple(phenotypes),))	
	if lives:
		if len(lives) == 1:
			where_clause.append("Live = %d" % lives[0])
		else:
			where_clause.append("Live In %s" % (tuple(lives),))	
	# Because we only label and care about the objects originate from T0
	where_clause.append("ValidFromT0 = 1")

	if where_clause:
		where_clause = "WHERE " + " AND ".join(where_clause) # Leave space after WHERE
	else: # If None output the whole database
		where_clause = ""

	conn = lite.connect(db_arg)
	with conn:
		cur = conn.cursor()
		# substr(WellID,1, 1), abs(substr(WellID, 2)) to get better order of WellID. Since simple order by WellID will make A12 comes before A2
		cur.execute("SELECT * FROM %s %s ORDER BY ExperimentName, substr(WellID,1, 1), abs(substr(WellID, 2)), ObjectID, Channel, NumOfHours" % (table_arg, where_clause))
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
	parser.add_argument("table_name")
	parser.add_argument("user_name")
	parser.add_argument("--experiment_name")
	parser.add_argument("--well_id")
	parser.add_argument("--obj_id")
	parser.add_argument("--channel")
	parser.add_argument("--phenotype")
	parser.add_argument("--live")
	parser.add_argument("output_csv")
	args = parser.parse_args()
	# Check if the table name is plain table name text or from history dataset text file
	if os.path.exists(args.table_name):
		with open(args.table_name, 'rb') as f:
			st = f.read()
			tb_name = st.split()[-1]
	else:
		tb_name = args.table_name		
	user_name = args.user_name
	if args.experiment_name:
		experiment_names = [e.strip() for e in args.experiment_name.split(',')	if e.strip() != '']
	else:
		experiment_names = None
	if args.well_id:
		well_ids = [w.strip() for w in args.well_id.split(',') if w.strip() != '']
	else:
		well_ids = None
	if args.obj_id:
		obj_ids = [int(o.strip()) for o in args.obj_id.split(',') if o.strip() != '']
	else:
		obj_ids = None
	if args.channel:
		channels = [c.strip() for c in args.channel.split(',') if c.strip() != '']
	else:
		channels = None
	if args.phenotype:
		phenotypes = [int(p.strip()) for p in args.phenotype.split(',') if p.strip() != '']
	else:
		phenotypes = None
	if args.live:
		lives = [int(l.strip()) for l in args.live.split(',') if l.strip() != '']
	else:
		lives = None


	# with open("/Users/guangzhili/Downloads/yyyy.txt", 'wb') as f:
	# 	f.write("%s, %s, %s, %s, %s, %s, %s, %s, %s\n %s, %s, %s, %s, %s, %s, %s, %s, %s" %(tb_name, args.user_name, args.experiment_name, args.well_id, args.obj_id, args.channel, args.phenotype, args.live, args.output_csv, type(tb_name), type(args.user_name), type(args.experiment_name), type(args.well_id), type(args.obj_id), type(args.channel), type(args.phenotype), type(args.live), type(args.output_csv)))
	# query_db_to_csv(DB_NAME, 'Images_A1', 'li', None, None, None, None, None, None, '/Users/guangzhili/Downloads/sss.txt')		
	query_db_to_csv(DB_NAME, tb_name, user_name, experiment_names, well_ids, obj_ids, channels, phenotypes, lives, args.output_csv)



	