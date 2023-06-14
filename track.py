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
import importlib

#Get Job ID
job_id=sys.argv[1]

#Get config
config_path=sys.argv[2]
spec = importlib.util.spec_from_file_location('TrackConfig', config_path)
modulevar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulevar)
config = modulevar.TrackConfig()

#Data folder
data_dir=sys.argv[3]
if os.path.isdir(data_dir) == False:
            logger.error(f'Incorrect data path specified: {data_dir}')
            exit(1)
else:
    data_dir=os.path.join(data_dir, config.MODEL_NAME, '')
    logger.info(f'Data dir: {data_dir}')

def main():
    #Get data
    logger.info('Get Orgasegment Results')
    results = pd.read_csv(f'{data_dir}results.csv')

    #Enrich data
    logger.info(f'Used regex: {regex}')
    results['well'] = results['name'].apply(lambda x: re.search(config.REGEX, x).group('WELL'))
    results['t'] = results['name'].apply(lambda x: re.search(config.REGEX, x).group('T'))
    
    ## Calculate centers and track organoids over time
    logger.info('Start tracking organoids')

    results['x'] = (results['x2'] + results['x1']) / 2
    results['y'] = (results['y2'] + results['y1']) / 2
    results = results.groupby('well').apply(tp.link, search_range=config.SEARCH_RANGE, memory=config.MEMORY, t_column='t').reset_index(drop=True)
    
    #Save results
    results.to_csv(f'{data_dir}tracked.csv', index=False)
        
if __name__ == "__main__":
    logger.info('Start tracking job...')
    main()
    logger.info('Tracking job completed!')
    ##Copy logging to data dir
    shutil.copy(f'log/JobName.{job_id}.out', f'{data_dir}/JobName.{job_id}.out')
    shutil.copy(f'log/JobName.{job_id}.err', f'{data_dir}/JobName.{job_id}.err')