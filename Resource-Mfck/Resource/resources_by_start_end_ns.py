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
import pandas as pd

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





def collect_metrics(start_time, end_time, metric, namespace):
    # Connect to Prometheus
    #prom = PrometheusConnect(url=prom, disable_ssl=True)
    
    # Convert string timestamps to datetime objects
    #start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    #end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    
    # Collect CPU metrics
    metric_query = f'sum(rate({metric}{{namespace="{namespace}", container!="POD", container!=""}}[1m])) by (pod,instance)'
    metric_data = prom.custom_query_range(
        query=metric_query,
        start_time=start_time,
        end_time=end_time,
        step="1s"
    )
    metric_df = MetricRangeDataFrame(metric_data)
    #print(metric_df.head(10))
    
    metric_df = metric_df[['instance','pod', 'value']].rename(columns={'value': metric})
    return metric_df


def collect_metrics_net(start_time, end_time, metric, namespace): # Connect to Prometheus #prom = PrometheusConnect(url=prom, disable_ssl=True)

    # Convert string timestamps to datetime objects
    #start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    #end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    # Collect CPU metrics
    net_query = f'sum(rate({metric}{{namespace="{namespace}"}}[1m])) by (pod,instance)'
    net_data = prom.custom_query_range(
        query=net_query,
        start_time=start_time,
        end_time=end_time,
        step="1s"
    )
    net_df = MetricRangeDataFrame(net_data)
    net_df = net_df[['instance','pod', 'value']].rename(columns={'value': metric})
    return net_df



#unique_nodes = container_cpu_usage_seconds_total['instance'].unique()
#print(unique_nodes)



def gen_metric(metric,ns, start,end,freq):
    #To avoid getting duplicate data, we beed separate query for each response code
    #a=prom.custom_query_range(metric+'{namespace="ob",}',start, end,freq)
    a=prom.custom_query_range(metric+'{namespace="'+ns+'",}',start, end,freq)
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
    return metric_df



def collect_resource_consumption(ns: str, csv_file_resource: str, csv_file_baro: str, start: datetime, end: datetime):
    container_cpu_usage_seconds_total=collect_metrics(start,end,'container_cpu_usage_seconds_total', ns)
    container_cpu_system_seconds_total=collect_metrics(start,end,'container_cpu_system_seconds_total', ns)
    container_memory_working_set_bytes=collect_metrics(start,end,'container_memory_working_set_bytes',ns)
    container_memory_rss=collect_metrics(start,end,'container_memory_rss', ns)
    container_network_receive_bytes_total=collect_metrics_net(start,end,'container_network_receive_bytes_total', ns)
    container_network_transmit_packets_total=collect_metrics_net(start,end,'container_network_transmit_packets_total', ns)
    
    cpu = pd.merge(container_cpu_usage_seconds_total, container_cpu_system_seconds_total,on=['timestamp','pod','instance'],how='inner') #suffixes=('_cpu', '_mem1'))
    mem = pd.merge(container_memory_working_set_bytes, container_memory_rss,on=['timestamp','pod','instance'],how='inner') #suffixes=('', '_mem2'))
    net = pd.merge(container_network_receive_bytes_total, container_network_transmit_packets_total, on=['timestamp','pod','instance'],how='inner')
    
    cpumem = pd.merge(cpu, mem,on=['timestamp','pod','instance'],how='inner')#suffixes=('_cpu', '_mem1'))
    cpumemnet = pd.merge(cpumem, net,on=['timestamp','pod','instance'],how='inner')#suffixes=('_cpu', '_mem1'))
    cpumemnet.to_csv(csv_file_resource, index=True)
                       

         
    

    # Let's assume your original DataFrame is called df
    # and timestamp is already set as the index

    metrics = ['container_cpu_usage_seconds_total', 'container_memory_working_set_bytes', 'container_network_transmit_packets_total']

    # Move the index back to a column temporarily
    cpumemnet = cpumemnet.reset_index()

    # Add a 'deployed_at' column (duplicate of node_name)
    cpumemnet['deployed_at'] = cpumemnet['instance']

    # Melt the DataFrame
    df_melted = pd.melt(cpumemnet, id_vars=['timestamp', 'pod'], value_vars=metrics + ['deployed_at'],   var_name='metric',  value_name='value')
                    
    # Create new column names like pod-metric
    df_melted['column'] = df_melted['pod'] + '_' + df_melted['metric']      

    # Pivot to wide format (so each pod and metric combination gets its own column)
    df_pivot = df_melted.pivot_table(index='timestamp',   columns='column',   values='value', aggfunc='first').reset_index()
    columns = ['timestamp'] + sorted([col for col in df_pivot.columns if col != 'timestamp'])
    # Sort the columns alphabetically except for timestamp
    df_pivot = df_pivot[columns]
    # Print the final DataFrame
    print(df_pivot)
    df_pivot.to_csv(csv_file_baro, index=True)                         
                                                            
    ''''#mergedData0=gen_metric('container_cpu_system_seconds_total',start,end,1)
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
    parser = argparse.ArgumentParser(description="Resource consumption  for Pods to CSV.")
    parser.add_argument('--namespace', type=str, required=True, help='Kubernetes namespace')
    parser.add_argument('--start', type=str, required=True, help='Start time (ISO format, e.g., 2025-04-15T10:00:00Z)')
    parser.add_argument('--end', type=str, required=True, help='End time (ISO format, e.g., 2025-04-15T14:00:00Z)')
    parser.add_argument('--file_resource', type=str, default='resources_consumption.csv', help='Output CSV file path')
    parser.add_argument('--file_baro', type=str, default='resources_baro_format.csv', help='Output CSV file path')
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
        csv_file_resource=args.file_resource,
        csv_file_baro=args.file_baro,
        start=start_time,
        end=end_time

    )    

    
