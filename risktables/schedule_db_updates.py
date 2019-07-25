'''
Created on Feb 17, 2019

schedule daily updates to database from yahoo

@author: bperlman1
'''
import sys
from distutils.command.build import build
if  not './' in sys.path:
    sys.path.append('./')
if  not '../' in sys.path:
    sys.path.append('../')
from risktables import schedule_it
from risktables import build_history
import argparse as ap
import time


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--hour',type=int,help='hour to start database update',default='1')
    parser.add_argument('--username',type=str,
                    help='username (None will be postgres)',
                    nargs='?')
    parser.add_argument('--password',type=str,
                    help='password (None will be blank)',
                    nargs='?')
    parser.add_argument('--days_to_fetch',type=int,
                    help='days to fetch for the update.  These days will first be deleted before the update (default = 4)',
                    default=4)
    args = parser.parse_args()
    h = args.hour
    logger = schedule_it.init_root_logger("logfile.log", "INFO")
    while True:
        logger.info(f"scheduling update for hour {h}")
        sch = schedule_it.ScheduleNext('hour', h,logger = logger)
        sch.wait()
        logger.info(f"updating history")
        bh = build_history.HistoryBuilder(update_table=True,username=args.username,password=args.password,logger=logger,days_to_fetch=args.days_to_fetch)
        bh.execute()
#         bh.update_yahoo_daily()
        logger.info(f"sleeping for an hour before next scheduling")
        time.sleep(60*60)
        