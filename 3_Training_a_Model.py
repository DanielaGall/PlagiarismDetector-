
# coding: utf-8

# In[ ]:


},
  {
   "cell_type": "code",
      "execution_count": 100,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# uncomment and fill in the line below!\n",
       "estimator.delete_endpoint()\n"
   ]
  },
  },
  {
   "cell_type": "code",
       "execution_count": 101,
   "metadata": {
    "collapsed": true
   },
      "outputs": [
    {
     "data": {
      "text/plain": [
       "[{'ResponseMetadata': {'RequestId': '19B9D8B942C63F1A',\n",
       "   'HostId': 'upUrHw4MicOCUIoFbc37Ji8Y5BLe16OTanw/XKZbHWh7ajh0jQdcF0sLZxFIKkgsl6/2IytCdW8=',\n",
       "   'HTTPStatusCode': 200,\n",
       "   'HTTPHeaders': {'x-amz-id-2': 'upUrHw4MicOCUIoFbc37Ji8Y5BLe16OTanw/XKZbHWh7ajh0jQdcF0sLZxFIKkgsl6/2IytCdW8=',\n",
       "    'x-amz-request-id': '19B9D8B942C63F1A',\n",
       "    'date': 'Mon, 22 Jul 2019 19:55:38 GMT',\n",
       "    'connection': 'close',\n",
       "    'content-type': 'application/xml',\n",
       "    'transfer-encoding': 'chunked',\n",
       "    'server': 'AmazonS3'},\n",
       "   'RetryAttempts': 0},\n",
       "  'Deleted': [{'Key': 'sagemaker-scikit-learn-2019-07-22-03-20-17-725/source/sourcedir.tar.gz'},\n",
       "   {'Key': 'sagemaker-scikit-learn-2019-07-22-04-17-59-471/source/sourcedir.tar.gz'},\n",
       "   {'Key': 'sagemaker-record-sets/LinearLearner-2019-07-22-19-20-28-678/matrix_0.pbr'},\n",
       "   {'Key': 'plagiarism/.ipynb_checkpoints/train-checkpoint.csv'},\n",
       "   {'Key': 'sagemaker-scikit-learn-2019-07-22-03-35-05-405/source/sourcedir.tar.gz'},\n",
       "   {'Key': 'plagiarism/linear-learner-2019-07-22-19-20-42-562/output/model.tar.gz'},\n",
       "   {'Key': 'plagiarism/.ipynb_checkpoints/test-checkpoint.csv'},\n",
       "   {'Key': 'sagemaker-scikit-learn-2019-07-22-03-24-16-941/source/sourcedir.tar.gz'},\n",
       "   {'Key': 'sagemaker-scikit-learn-2019-07-22-03-24-36-715/source/sourcedir.tar.gz'},\n",
       "   {'Key': 'sagemaker-scikit-learn-2019-07-22-04-06-05-126/source/sourcedir.tar.gz'},\n",
       "   {'Key': 'plagiarism/test.csv'},\n",
       "   {'Key': 'sagemaker-record-sets/LinearLearner-2019-07-22-19-20-28-678/.amazon.manifest'},\n",
       "   {'Key': 'sagemaker-scikit-learn-2019-07-22-03-21-57-277/source/sourcedir.tar.gz'},\n",
       "   {'Key': 'plagiarism/train.csv'},\n",
       "   {'Key': 'sagemaker-scikit-learn-2019-07-22-03-31-28-689/source/sourcedir.tar.gz'}]}]"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# deleting bucket, uncomment lines below\n",
    "\n",
       "bucket_to_delete = boto3.resource('s3').Bucket(bucket)\n",
    "bucket_to_delete.objects.all().delete()"
   ]
  },
  {
      
      

