
# coding: utf-8

# In[ ]:


},
 {
  "cell_type": "code",
     "execution_count": 393,
  "metadata": {},
  "outputs": [],
  "source": [
   "#pd.concat([pd.DataFrame(test_y), pd.DataFrame(test_x)], axis=1)"
  ]
 },
 {
  "cell_type": "code",
  "execution_count": 399,
  "metadata": {
   "collapsed": true
  },
     os.makedirs(data_dir)\n",
   "    \n",
   "    \n",
     # combine data and sent to csv\n",
   "    pd.concat([pd.DataFrame(y), pd.DataFrame(x)], axis=1) \\\n",
   "        .to_csv(os.path.join(data_dir, filename), header=False, index=False)\n",
   "    \n",
   "    # nothing is returned, but a print statement indicates that the function has run\n",
   "    print('Path created: '+str(data_dir)+'/'+str(filename))"
     },
 {
  "cell_type": "code",
      "execution_count": 400,
  "metadata": {
   "collapsed": true
  },
      "outputs": [
   {
    "name": "stdout",
    "output_type": "stream",
    "text": [
     "Path created: test_csv/to_delete.csv\n",
     "Tests passed!\n"
    ]
   }
  ],
  "source": [
   "\"\"\"\n",
   "DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE\n",
      },
 {
  "cell_type": "code",
     "execution_count": 401,
  "metadata": {
   "collapsed": true
  },
     },
 {
  "cell_type": "code",
     "execution_count": 402,
  "metadata": {
   "collapsed": true
  },
     "outputs": [
   {
    "name": "stdout",
    "output_type": "stream",
    "text": [
     "Path created: plagiarism_data/train.csv\n",
     "Path created: plagiarism_data/test.csv\n"
    ]
   }
  ],
  "source": [
   "# can change directory, if you want\n",
   "data_dir = 'plagiarism_data'\n",
      
      
      
      
      
      
      

