# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd
import time
import catboost_model as catb

# Main
if __name__  == "__main__":    
    start = time.time()
    
    # Load train and test datasets
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    #Catboost handles Null values and Categorical variables
    clf = catb.CatbClassifier(train_df)    
    
    #Create features and target datasets
    clf.generate_x_y('Survived')    
 
    #Train the model
    accuracy = clf.model_train()
    print('Best model validation accuracy: {:.4}'.format(accuracy))       
 
    #Cross validation
    cv_acc_score = clf.model_cross_validation()        
    print('Cross validation Accuracy : {:.4}'.format(cv_acc_score))       
    
    #Select best model for prediction 
    clf.save_model('saved_model')
    res = clf.model_predict(test_df)

    submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':res})  
    submission.to_csv('results.csv',index=False)
    
    end = time.time()
    print(f"Time taken for prediction : {.4} s".format(end-start))       
    
    # Delete variables
    del [train_df, test_df,accuracy,cv_acc_score,res,start,end]
    
