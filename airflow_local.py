# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:54:53 2020

@author: adityaroyal
"""
from datetime import timedelta

import airflow
from airflow import DAG
from airflow.operators.papermill_operator import PapermillOperator


default_args = {
    'owner': 'airflow',    
    'start_date': airflow.utils.dates.days_ago(2),
    # 'end_date': datetime(2018, 12, 30),
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    # If a task fails, retry it once after waiting
    # at least 5 minutes
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    }
dag = DAG(
    'diabetes',
    default_args=default_args,
    description='A simple tutorial DAG',
    # Continue to run DAG once per day
    schedule_interval=timedelta(days=1),
)



t1 = PapermillOperator(
    task_id="preprocessing_notebook",
    input_nb="/home/adi/airflow/dags/preprocessing--sklearn (1).ipynb",
    
    parameters={"msgs": "Ran from Airflow at {{ execution_date }}!"},
    dag=dag
)

t2 = PapermillOperator(
    task_id="train_notebook",
    input_nb="/home/adi/airflow/dags/train_sklearn (1).ipynb",
    
    parameters={"msgs": "Ran from Airflow at {{ execution_date }}!"},
    dag=dag
)


t3 = PapermillOperator(
    task_id="test_notebook",
    input_nb="/home/adi/airflow/dags/test_sklearn (1).ipynb",
    
    parameters={"msgs": "Ran from Airflow at {{ execution_date }}!"},
    dag=dag
)



t2.set_upstream(t1)
t3.set_upstream(t2)