import pandas as pd
import numpy as np

class NB:
    def __init__(self, target, dataframe):
        self.df = dataframe
        # Target/Category Column
        self.c_n = target
        # Column Names
        self.cols = list(self.df.columns)
        self.cols.remove(self.c_n)
        
        self.store = {}
        self.likelihood_for_all_()
        
    def likelihood_cal(self, x, y, z):
        """ 
        x -> Column Name (String)
        y -> Column Value (String)
        z -> Class value (String)
        c_n -> Class Name (Target)
        
        Returns -> P(x = y | c_n = z)
        """
        df = self.df
        
        if x not in self.cols:
            raise KeyError("Feature(column) not present in the Training Dataset")
        
        res =  len(df[(df[x] == y) & (df[self.c_n] == z)]) /len(df[df[self.c_n] == z])
        
        if res == 0.0:
            return 1/(len(df[df[self.c_n] == z]) + len(df[x].unique()))
        
        return res
    
    def likelihood_for_all_(self):     
        df = self.df
        
        dict1 = {}
        for x in self.cols:
            dict2 = {}
            for y in df[x].unique():
                dict3 = {}
                for z in df[self.c_n].unique():
                    #print('P({}="{}"|{}="{}") = {}'.format(x,y,self.c_n,z,self.likelihood_cal(x, y, z)))
                    dict3[z] = self.likelihood_cal(x, y, z)
                dict2[y] = dict3
            dict1[x] = dict2
        
        self.store = dict1
    
    def likelihood_expr(self, class_val, expr):
        val = 1  
        
        for k,v in expr:
            try:
                store_val = self.store[k][v][class_val]
            except:
                store_val = self.likelihood_cal(k,v,class_val)
                
            val *= store_val
                                         
        return val
    
    def prior(self, class_val):
        df = self.df
        return len(df[df[self.c_n] == class_val])/df.shape[0]
    
    def predict(self, X):
        df = self.df
        
        if type(X) == pd.core.series.Series:
            values_list = [list(X.items())]
            
        elif type(X) == pd.core.frame.DataFrame:
            values_list = [list(y.items()) for x,y in X.iterrows()]
            
        else:
            raise TypeError('{} is not supported type'.format(type(X)))
            
        
        predictions_list = []
        for values in values_list:
            likelihood_priors = {}
            for class_val in df[self.c_n].unique():
                likelihood_priors[class_val] = self.prior(class_val)*self.likelihood_expr(class_val,values)
            #print(likelihood_priors)
            
            normalizing_prob = np.sum([x for x in likelihood_priors.values()])
            probabilities = [(y/normalizing_prob,x) for x,y in likelihood_priors.items()]
            #print(probabilities)
            
            if len(probabilities) == 2:
                # For 2 Class Predictions
                max_prob = max(probabilities)[1]
                predictions_list.append(max_prob)
            
            else:
                # For Mulit Class Predictions
                exp_1 = [np.exp(x) for x,y in probabilities]
                exp_2 = np.sum(exp_1)
                softmax = exp_1/exp_2
                #print(softmax)
                class_names = [y for x,y in probabilities]
                softmax_values = [(x,y) for x,y in zip(softmax,class_names)]
                #print(softmax_values)
                max_prob = max(softmax_values)[1]
                predictions_list.append(max_prob)
        
        
        return predictions_list
    
    def accuracy_score(self, X, Y):
        assert len(X) == len(Y), 'Given values are not equal in size'
        
        total_matching_values = [x == y for x,y in zip(X,Y)]
        return (np.sum(total_matching_values)/len(total_matching_values))*100
    
    def calculate_confusion_matrix(self, X, Y):
        df = self.df
        
        unique_class_values = df[self.c_n].unique()
        decimal_class_values = list(range(len(unique_class_values)))
        numerical = {x:y for x,y in zip(unique_class_values, decimal_class_values)}
        
        x = [numerical[x] for x in X]
        y = [numerical[y] for y in Y]
        
        
        n = len(decimal_class_values)
        confusion_matrix = np.zeros((n,n))
        
        for i,j in zip(x,y):
            if i == j:
                confusion_matrix[i][i] += 1
            elif i != j:
                confusion_matrix[i][j] += 1
        
        return confusion_matrix
            
    
    def precision_score(self, X, Y):
        assert len(X) == len(Y), 'Given values are not equal in size'
        
        confusion_matrix = self.calculate_confusion_matrix(X,Y)
        tp = confusion_matrix[0][0]
        fp = confusion_matrix[1][0]
        
        return tp / (tp+fp)
    
    def recall_score(self, X, Y):
        assert len(X) == len(Y), 'Given values are not equal in size'
        
        confusion_matrix = self.calculate_confusion_matrix(X,Y)
        tp = confusion_matrix[0][0]
        fn = confusion_matrix[0][1]
        
        return tp / (tp+fn)
                
                


if __name__ == "__main__":
    data = pd.read_csv('../dataset/mushrooms.csv')
    
    ind = list(data.index)
    np.random.shuffle(ind)
    
    # Train:Test = 75%:25%
    train_len = int(data.shape[0]*0.75)
    train_ind = ind[:train_len]
    training_data = data.iloc[train_ind,:]

    test_ind = ind[train_len:]
    testing_data = data.iloc[test_ind,:]

    print('Training_data size -> {}'.format(training_data.shape))
    print('Testing_data size -> {}'.format(testing_data.shape))

    assert data.shape[0] ==  len(train_ind)+ len(test_ind), 'Not equal distribution'
    
    genx = NB(target='class',dataframe=training_data)
    
    y_test = list(testing_data.iloc[:,0])
    y_pred = genx.predict(testing_data.iloc[:,1:])
    #print(y_test)
    #print(y_pred)

    print('Accuracy Score -> {} %'.format(round(genx.accuracy_score(y_test,y_pred),3)))
    print('Precison Score -> {}'.format(round(genx.precision_score(y_test,y_pred),3)))
    print('Recall Score -> {}'.format(round(genx.recall_score(y_test,y_pred),3)))

