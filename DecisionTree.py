import numpy as np

NULL = 0 
class bsearch(object):
    '''
    simple binary search tree, with public functions of search, insert and traversal
    '''
    def __init__ (self, value) :
        self.value = value
        self.left = self.right = NULL

    def search(self, value) :
        if self.value==value :
            return True 
        elif self.value>value :
            if self.left==NULL :
                return False 
            else:
                return self.left.search(value)
        else :		
            if self.right==NULL : 
                return False 
            else :
                return self.right.search(value)

    def insert(self, value) :
        if self.value==value :
            return False 
        elif self.value>value :
            if self.left==NULL :
                self.left = bsearch(value)
                return True 
            else :
                return self.left.insert(value)
        else :
            if self.right==NULL :
                self.right = bsearch(value)
                return True 
            else :
                return self.right.insert(value)

    def inorder(self)  :
        if self.left !=NULL  :
            self.left.inorder()
        if self != NULL : 
            print (self.value, " ", end="")
        if self.right != NULL : 
            self.right.inorder()


class TreeNode(object):
    '''
    A class for storing necessary information at each tree node.
    Nodes should be initialized as objects of this class. 
    '''
    def __init__(self, d=None, threshold=None, l_node=None, r_node=None, label=None, is_leaf=False, gini=None, n_samples=None):
        '''
        Input:
            d: index (zero-based) of the attribute/feature selected for splitting use. int
            threshold: the threshold for attribute d. If the attribute d of a sample is <= threshold, the sample goes to left 
                       branch; o/w right branch. float
            l_node: left children node/branch of current node. TreeNode
            r_node: right children node/branch of current node. TreeNode
            label: the most common label at current node. int/float
            is_leaf: True if this node is a leaf node; o/w False. bool
            gini: stores gini impurity at current node. float
            n_samples: number of training samples at current node. int
        '''
        self.d = d
        self.threshold = threshold
        self.l_node = l_node
        self.r_node = r_node
        self.label = label
        self.is_leaf = is_leaf
        self.gini = gini
        self.n_samples = n_samples


def load_data(fdir):
    '''
    Load attribute values and labels from a npy file. 
    Data is assumed to be stored of shape (N, D) where the first D-1 cols are attributes and the last col stores the labels.
    Input:
        fdir: file directory. str
    Output:
        data_x: feature vector. np ndarray
        data_y: label vector. np ndarray
    '''
    data = np.load(fdir)
    data_x = data[:, :-1]
    data_y = data[:, -1].astype(int)
    print(f"x: {data_x.shape}, y:{data_y.shape}")
    return data_x, data_y


class CART(object):
    '''
    Classification and Regression Tree (CART). 
    '''
    def __init__(self, max_depth=None):
        '''
        Input:
            max_depth: maximum depth allowed for the tree. int/None.
        Instance Variables:
            self.max_depth: stores the input max_depth. int/inf
            self.tree: stores the root of the tree. TreeNode object
        '''
        self.max_depth = float('inf') if max_depth is None else max_depth 
        self.tree = None        #root node

    def gini_Impure(self, X, y):
        uni_arr, counts = np.unique(y, return_counts=True)
        freq = counts / np.sum(counts)
        gini_I = 1 - np.sum(np.square(freq))
        return gini_I
        

    def Thresh_split(self, X, y, threshold, feature):
        X_left = X[X[:, feature] <= threshold]
        y_left = y[X[:, feature] <= threshold]
        X_right = X[X[:, feature] > threshold]
        y_right = y[X[:, feature] > threshold]

        l_gini = self.gini_Impure(X_left, y_left)
        r_gini = self.gini_Impure(X_right, y_right)

        return X_left, y_left, X_right, y_right, l_gini, r_gini


    def best_split(self, parent, train_X, train_y, depth_remain):

        l_node = None
        r_node = None
        best_gini_split = None
        best_l_split_X = None
        best_l_split_y = None
        best_r_split_X = None
        best_r_split_y = None

        parent.n_samples = train_X.shape[0]
        uni_arr, counts = np.unique(train_y, return_counts=True)
        parent.label = uni_arr[np.argmax(counts)].astype(int)

        if (depth_remain is 0) or (parent.gini is 0):
            parent.is_leaf = True
            return parent

        for feature_i, features in enumerate(train_X.T):
            uni_arr, counts = np.unique(features, return_counts=True)
            if (len(uni_arr) <= 1):
                continue
            for index in range(len(uni_arr)-1):
                threshold = (uni_arr[index] + uni_arr[index + 1]) / 2

                l_train_X, l_train_y, r_train_X, r_train_y, l_gini, r_gini = self.Thresh_split(train_X, train_y, threshold, feature_i)
                
                split_gini = l_gini * (l_train_X.shape[0] / parent.n_samples) + r_gini * (r_train_X.shape[0] / parent.n_samples)

                if (split_gini < parent.gini):
                    if (best_gini_split is None) or (split_gini < best_gini_split):
                        best_gini_split = split_gini
                        parent.d = feature_i
                        parent.threshold = threshold
                        best_l_split_X = l_train_X
                        best_l_split_y = l_train_y
                        best_r_split_X = r_train_X
                        best_r_split_y = r_train_y
                        l_node = TreeNode(gini=l_gini)
                        r_node = TreeNode(gini=r_gini)
                        
        if (best_gini_split is None):
            parent.is_leaf = True
            return parent
        
        parent.l_node = self.best_split(l_node, best_l_split_X, best_l_split_y, (depth_remain - 1))
        parent.r_node = self.best_split(r_node, best_r_split_X, best_r_split_y, (depth_remain - 1))

        return parent

    def train(self, X, y):
        '''
        Build the tree from root to all leaves. The implementation follows the pseudocode of CART algorithm.  
        Input:
            X: Feature vector of shape (N, D). N - number of training samples; D - number of features. np ndarray
            y: label vector of shape (N,). np ndarray
        '''
        
        gini = self.gini_Impure(X, y)
        root = TreeNode()
        root.gini = gini
        self.tree = self.best_split(root, X, y, self.max_depth)


    def accuracy(self, X_val, y_val):
        accuracy = np.mean(self.test(X_val) == y_val)
        return accuracy

    def traverse(self, node, feature):
        if (node.is_leaf):
            return node.label
        else:
            if (feature[node.d] <= node.threshold):
                return self.traverse(node.l_node, feature)
            else:
                return self.traverse(node.r_node, feature)
    
    def test(self, X_test):
        '''
        Predict labels of a batch of testing samples. 
        Input:
            X_test: testing feature vectors of shape (N, D). np array
        Output:
            prediction: label vector of shape (N,). np array, dtype=int
        '''

        pred = []
        for feature in X_test:
            pred.append(self.traverse(self.tree, feature))
        pred = np.array(pred)
        return pred



    def visualize_tree(self):
        '''
        A simple function for tree visualization. 
        '''

        print('ROOT: ')
        def print_tree(tree, indent='\t|', dict_tree={}, direct='L'):
            if tree.is_leaf == True:
                dict_tree = {direct: str(tree.label)}
            else:
                print(indent + 'attribute: %d/threshold: %.5f' % (tree.d, tree.threshold))

                if tree.l_node.is_leaf == True:
                    print(indent + 'L -> label: %d' % tree.l_node.label)
                else:
                    print(indent + "L -> ",)
                a = print_tree(tree.l_node, indent=indent + "\t|", direct='L')
                aa = a.copy()

                if tree.r_node.is_leaf == True:
                    print(indent + 'R -> label: %d' % tree.r_node.label)
                else:
                    print(indent + "R -> ",)
                b = print_tree(tree.r_node, indent=indent + "\t|", direct='R')
                bb = b.copy()

                aa.update(bb)
                stri = indent + 'attribute: %d/threshold: %.5f' % (tree.d, tree.threshold)
                if indent != '\t|':
                    dict_tree = {direct: {stri: aa}}
                else:
                    dict_tree = {stri: aa}
            return dict_tree
        try:
            if self.tree is None:
                raise RuntimeError('No tree has been trained!')
        except:
            raise RuntimeError('No self.tree variable!')
        _ = print_tree(self.tree)

        
def GridSearchCV(X, y, depth=[1, 40]):
    '''
    Grid search and cross validation.
    Input:
        X: full training dataset. Not split yet. np ndarray
        y: full training labels. Not split yet. np ndarray
        depth: [minimum depth to consider, maximum depth to consider]. list of integers
    Output:
        best_depth: the best max_depth value from grid search results. int
        best_acc: the validation accuracy corresponding to the best_depth. float
        best_tree: a decision tree object that is trained with 
                   full training dataset and best max_depth from grid search. instance
    '''
    
    depths = np.linspace(depth[0], depth[1], num=10, dtype=int)    

    best_depth = None
    best_acc = 0.0
    best_tree = None
    y = np.array([y]).T

    data = np.append(X, y, 1)
    np.random.shuffle(data)
    split1, split2, split3, split4, split5 = np.array_split(data, 5)

    for max_depth in depths:

        #5-fold cross validation
        for cv in range(5):
            if cv is 0:
                validation = split1
                training = np.append(split2, split3, 0) 
                training = np.append(training, split4, 0)
                training = np.append(training, split5, 0)
            elif cv is 1:
                validation = split2
                training = np.append(split1, split3, 0) 
                training = np.append(training, split4, 0)
                training = np.append(training, split5, 0)
            elif cv is 2:
                validation = split3
                training = np.append(split1, split2, 0) 
                training = np.append(training, split4, 0)
                training = np.append(training, split5, 0)
            elif cv is 3:
                validation = split4
                training = np.append(split1, split2, 0) 
                training = np.append(training, split3, 0)
                training = np.append(training, split5, 0)
            elif cv is 4:
                validation = split5
                training = np.append(split1, split2, 0) 
                training = np.append(training, split3, 0)
                training = np.append(training, split5, 0)
            
            X_train = training[:,:-1]
            y_train = np.array([training[:,-1]]).T
            X_val = validation[:,:-1]
            y_val = np.array([validation[:,-1]]).T

            tree = CART()
            tree.train(X_train, y_train)
            accuracy = tree.accuracy(X_val, y_val)

            if best_acc < accuracy:
                best_acc = accuracy
                best_depth = max_depth
    
    cart = CART(best_depth)
    cart.train(X,y)
    best_tree = cart

    return best_depth, best_acc, best_tree

# main

X_train, y_train = load_data('winequality-red-train.npy')
best_depth, best_acc, best_tree = GridSearchCV(X_train, y_train, [1, 40])
print('Best depth from 5-fold cross validation: %d' % best_depth)
print('Best validation accuracy: %.5f' % (best_acc))