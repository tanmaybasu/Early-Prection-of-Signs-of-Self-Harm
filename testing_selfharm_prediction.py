#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:14:36 2021

@author: Tanmay Basu
"""

from selfharm_prediction import selfharm_prediction


clf=selfharm_prediction('/home/tanmay/erisk2021/',model='entropy',model_path='saved_models/entropy_svm/',clf_opt='s',no_of_selected_terms=3000,output_file='output/entropy_svm_phase11.json')
#clf=selfharm_prediction('/home/tanmay/erisk2021/',model='entropy',model_path='saved_models/entropy_rf/',clf_opt='r',no_of_selected_terms=3000,output_file='output/entropy_rf_phase11.json')
#clf=selfharm_prediction('/home/tanmay/erisk2021/',model='tfidf',clf_opt='s',model_path='saved_models/tfidf_svm/',no_of_selected_terms=2500,output_file='output/tfidf_svm_phase11.json')
#clf=selfharm_prediction('/home/tanmay/erisk2021/',model='doc2vec',model_path='saved_models/doc2vec_ab/',vec_len=50,clf_opt='a',output_file='output/doc2vec_ab_phase11.json')
#clf=selfharm_prediction('/home/tanmay/erisk2021/',model='doc2vec',model_path='saved_models/doc2vec_rf/',vec_len=50,clf_opt='r',output_file='output/doc2vec_rf_phase11.json')

#clf=selfharm_prediction('/Users/basut/erisk2021/',model='bert',model_source='bert-base-uncased',output_file='output/bert_phase11.json')

clf.selfharm_prediction()

