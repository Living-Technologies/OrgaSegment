#Init logger
import logging
import logging.config
logging.config.fileConfig('conf/logging.conf')
logger = logging.getLogger(__file__)

#Import packages
import sys
import shutil
import pandas as pd
import trackpy as tp
import re
import os
import json

#Get Job ID
job_id=sys.argv[1]

#Data folder
data_dir=sys.argv[2]
if os.path.isdir(data_dir) == False:
            logger.error(f'Incorrect data path specified: {data_dir}')
            exit(1)
else:
    data_dir=os.path.join(data_dir, 'orgasegment', '')
    logger.info(f'Data dir: {data_dir}')

def main():
    #Load config
    config = f'{data_dir}track-config.json'
    if os.path.isfile(config):
        config = json.load(open(config))
        regex = config['regex']
    
        #Get data
        logger.info('Get Orgasegment Results')
        results = pd.read_csv(f'{data_dir}results.csv')

        #Enrich data
        logger.info(f'Used regex: {regex}')
        results['well'] = results['name'].apply(lambda x: re.search(regex, x).group('WELL'))
        results['t'] = results['name'].apply(lambda x: re.search(regex, x).group('T'))
        
        ## Calculate centers and track organoids over time
        logger.info('Start tracking organoids')

        results['x'] = (results['x2'] + results['x1']) / 2
        results['y'] = (results['y2'] + results['y1']) / 2
        results = results.groupby('well').apply(tp.link, search_range=50, memory=2, t_column='t').reset_index(drop=True)
        
        ## Filter out particles that are seen less than 50% of the time
        #results = results[results.groupby(['well','particle'])['t'].transform('count') >= (max(results['t']) / 2)]
        
        #Save results
        results.to_csv(f'{data_dir}tracked.csv', index=False)
    
    else:
        logger.warning(f'No tracking config detected. Results in {data_dir} will not be tracked')
        exit(0) 
        
if __name__ == "__main__":
    logger.info('Start tracking job...')
    main()
    logger.info('Tracking job completed!')
    ##Copy logging to data dir
    shutil.copy(f'log/JobName.{job_id}.out', f'{data_dir}/JobName.{job_id}.out')
    shutil.copy(f'log/JobName.{job_id}.err', f'{data_dir}/JobName.{job_id}.err')