import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt
#import networkx as nx
#import raphtory as rp
import datetime
from dateutil.relativedelta import relativedelta
from prometheus_api_client import PrometheusConnect, MetricsList, Metric, MetricSnapshotDataFrame, MetricRangeDataFrame


import os
import csv
from datetime import datetime
from kubernetes import client, config
import pytz
import datetime
from pytz import timezone
import argparse


#prom = PrometheusConnect(url ="http://172.189.51.189:80", disable_ssl=True)
prom = PrometheusConnect(url ="http://4.178.243.119:9090", disable_ssl=True)
#prom = PrometheusConnect(url ="http://raphtory03:9090", disable_ssl=True)
#prom = PrometheusConnect(url ="http://localhost:8080", disable_ssl=True)
#prom = PrometheusConnect(url ="http://raphtory03:10000", disable_ssl=True)


'''

ns="ob"
start="2025-04-16 19:30:00"
end="2025-04-16 19:50:00"
start = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
end = datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
print("Start: ", start, "End: ", end)'''


def gen_metric_all(metric,ns,start,end,freq):
    #a=prom.custom_query_range(metric+'{namespace="ob", reporter="destination", request_protocol="http" , response_code="200",}',start, end,freq)
    a=prom.custom_query_range(metric+'{namespace="'+ns+'", reporter="destination",}',start, end,freq)
    #a=prom.custom_query_range(metric+'{namespace="'+ns+'",}',start, end, freq)
    metric_df9  = MetricRangeDataFrame(a)
    data=metric_df9[['source_workload', 'destination_workload','request_protocol','response_flags','reporter','response_code','grpc_response_status','value']]
    data=data.sort_values(by='timestamp')
    data = data[data['source_workload'] != "unknown"]

    # Get all unique values from the response_code column
    unique_response_codes = data['response_code'].unique()
    print(unique_response_codes)

    # Get all unique values from the grpc_response_status column
    unique_grpc_response_codes = data['grpc_response_status'].unique()
    print(unique_grpc_response_codes)


    # Get all unique values from the response_flags column
    unique_response_flags = data['response_flags'].unique()
    print(unique_response_flags)

    # Get all unique values from the request_protocol column
    unique_request_protocol = data['request_protocol'].unique()
    print(unique_request_protocol)
    return data



'''def gen_metric(metric,ns, start,end,freq):
    #To avoid getting duplicate data, we beed separate query for each response code
    #a=prom.custom_query_range(metric+'{namespace="ob",}',start, end,freq)
    a=prom.custom_query_range(metric+'{namespace="'+ns+'",}',start, end, freq)
    #print(a)
    metric_df  = MetricRangeDataFrame(a)
    metric_df=metric_df[['instance', 'pod', 'container','value']]
    #metric_df = metric_df[metric_df['container'] != '']
    metric_df = metric_df[metric_df['container'].notna()]
    metric_df=metric_df.drop_duplicates()
    metric_df = metric_df.drop(metric_df[metric_df['container'] == 'istio-proxy'].index)
    metric_df = metric_df[~metric_df['pod'].str.startswith('load')]
    metric_df = metric_df[~metric_df['pod'].str.startswith('redis')]
    #metric_df = metric_df[~metric_df['pod'].str.startswith('front')]
    
    metric_df.rename(columns={'value':metric}, inplace=True)
    #metric_df = metric_df.sort_values(by=['timestamp', 'pod'], ascending=[True, True], na_position='last')
    metric_df = metric_df.sort_values(by=['pod','timestamp'], ascending=[True, True], na_position='last')

    #mergedData0.to_csv(name, index=True)
    return metric_df'''



def collect_resource_consumption(ns: str, csv_file: str, start: datetime, end: datetime):
    #Merge all files
    a=gen_metric_all('istio_requests_total', ns, start,end,1)
    a.columns = ['source_workload', 'destination_workload','request_protocol','response_flags','reporter','response_code','grpc_response_status','total_request']
    b=gen_metric_all('istio_request_bytes_sum',ns, start, end,1)
    b.columns = ['source_workload', 'destination_workload','request_protocol','response_flags','reporter','response_code','grpc_response_status','istio_request_bytes_sum']
    c=gen_metric_all('istio_request_duration_milliseconds_sum',ns, start ,end,1)
    c.columns = ['source_workload', 'destination_workload','request_protocol','response_flags','reporter','response_code','grpc_response_status','istio_request_duration_milliseconds_sum']
    
    mergedData1 = pd.merge(a, b, on=['timestamp','source_workload', 'destination_workload','request_protocol','response_flags','reporter','response_code','grpc_response_status'], how='inner')
    mergedData1.to_csv('istio_request_1.csv', index=True)
    mergedData2 = pd.merge(mergedData1, c, on=['timestamp','source_workload', 'destination_workload','request_protocol','response_flags','reporter','response_code','grpc_response_status'], how='inner')


    mergedData2 = mergedData2.loc[:,~mergedData2.columns.duplicated()].copy()
    mergedData2.to_csv(csv_file, index=True)
    #mergedData2.to_csv('istio_request_v3.csv', index=True)




    '''#mergedData0=gen_metric('container_cpu_system_seconds_total',start,end,1)
    mergedData0=gen_metric('container_cpu_usage_seconds_total',ns,start,end,1)
    #mergedData0.to_csv('container_cpu_usage_seconds_total.csv', index=True)
    mergedData0.head(20)


    mergedData1=gen_metric('container_cpu_system_seconds_total',ns,start,end,1)
    #mergedData1.to_csv('container_cpu_system_seconds_total.csv', index=True)
    mergedData1.head(30)



    mergedData2=gen_metric('container_memory_working_set_bytes',ns,start,end,1)
    #mergedData2.to_csv('container_memory_working_set_bytes.csv', index=True)
    mergedData2.head(30)


    mergedData3=gen_metric('container_memory_rss',ns,start,end,1)
#   mergedData3.to_csv('container_memory_rss.csv', index=True)
    mergedData3.head(30)


    # Merge with explicit suffixes for overlapping columns
    mergedData10 = pd.merge(mergedData0, mergedData1,
                       on=['timestamp','instance','pod','container'],
                       how='inner')
                       #suffixes=('_cpu', '_mem1'))

    mergedData11 = pd.merge(mergedData10, mergedData2,
                       on=['timestamp','instance','pod','container'],
                       how='inner')
                       #suffixes=('', '_mem2'))

    mergedData12 = pd.merge(mergedData11, mergedData3,
                       on=['timestamp','instance','pod','container'],
                       how='inner')



    mergedData12.to_csv(csv_file, index=True)
    mergedData12.head(10)'''




def parse_args():
    parser = argparse.ArgumentParser(description="Communication for Pods to CSV.")
    parser.add_argument('--namespace', type=str, required=True, help='Kubernetes namespace')
    parser.add_argument('--start', type=str, required=True, help='Start time (ISO format, e.g., 2025-04-15T10:00:00Z)')
    parser.add_argument('--end', type=str, required=True, help='End time (ISO format, e.g., 2025-04-15T14:00:00Z)')
    parser.add_argument('--output', type=str, default='pods_communication.csv', help='Output CSV file path')
    return parser.parse_args()

def iso_to_utc(iso_string):
    return datetime.fromisoformat(iso_string.replace("Z", "+00:00"))

if __name__ == "__main__":
    args = parse_args()
    start_time = datetime.datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
    
    start_time= start_time.astimezone(timezone('UTC'))
    end_time= end_time.astimezone(timezone('UTC'))
    
    collect_resource_consumption(
        ns=args.namespace,
        csv_file=args.output,
        start=start_time,
        end=end_time

    )    

    
