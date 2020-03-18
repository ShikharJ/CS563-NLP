import numpy as np

class Node():
    cnt = 0
    def __init__(self, ):
        self.leaf = False
        self.majority_class = None
        self.attribute_index = None
        self.children = dict() # key: attribute_value, value: child_node
        self.sent_len_split_val = None # Used at inference time, if attribute_index is 0
        self.id = Node.cnt
        Node.cnt+=1
    
    def __str__(self,):
        return 'ID: {}  isLeaf: {} majority: {} split_idx: {} split_val = {}'.format(self.id, 
                                                                                    self.leaf, 
                                                                                    self.majority_class, 
                                                                                    self.attribute_index, 
                                                                                    list(self.children.keys())
                                                                                   )
    def __repr__(self):
        return str(self)
    
    def traverse_print(self,):
        print(self)
        for _, child in self.children:
              child.traverse_print()

class DecisionTree():
   
    def __init__(self):
        self.root = None
        return
   
    @staticmethod
    def compute_entropy(labels):
        entropy = 0.0
        totSamples = len(labels)
        labelSet = set(labels.reshape(-1))
        for label in labelSet:
            prob = np.sum(labels == label) / totSamples
            if prob > 1e-12:
                entropy -= np.log(prob) * prob
            
        return entropy
 
   
    @staticmethod
    def compute_dict_entropy(attr_count):
        entropy = 0
        totSamples = sum(attr_count.values())
       
        labelSet = attr_count.keys()
        for label in labelSet:
            prob = attr_count[label] / totSamples
            if prob > 1e-12:
                entropy -= np.log(prob) * prob
        return entropy
   
    def split_node(self, parent, data, labels, used_attr_index):
        num_instances = data.shape[0]
        parent_info = self.compute_entropy(labels) * num_instances
       
        parent.majority_class = Counter(labels.reshape(-1)).most_common(1)[0][0]
               
        if parent_info == 0 :
            parent.leaf = True
       
        best_attr_index = None
        best_info_gain = -float('inf')
        best_gain_ratio = -float('inf')
        best_attr_keys = None
       
        # sent length case special
        attr_split_info = 0
        attr_count = dict()
        sent_len_split_val = stats.mode(data[:, 0])[0][0]
        le_ids = np.where(data[:, 0] <= sent_len_split_val)[0]
        gt_ids = np.where(data[:, 0] > sent_len_split_val)[0]
        attr_count[0] = le_ids.shape[0]
        attr_count[1] = gt_ids.shape[0]
        attr_split_info = (attr_count[0] * self.compute_entropy(labels[le_ids])) + (attr_count[1] * self.compute_entropy(labels[gt_ids]) )    
        attr_gain = parent_info - attr_split_info
        attr_gain_ratio = self.compute_dict_entropy(attr_count) * attr_gain
        if best_gain_ratio < attr_gain_ratio and  attr_gain_ratio > 0 :
                best_attr_index = 0
                best_info_gain = attr_gain
                best_gain_ratio = attr_gain_ratio
                best_attr_keys = attr_count.keys()
           
        # during ablation, sentence length can be initialized to all zeros this will prevent splittiung in sent dimension/.
        for i in range(1, data.shape[1]): # starts from 1 as zero is sentence length (always.) .
            if i in used_attr_index:
                continue
            attr_split_info = 0
            attr_count = dict()
            for attr_val in set(data[:, i].reshape(-1)):
                ids = np.where(data[:, i] == attr_val)[0]
                attr_count[attr_val] = len(ids)
                attr_split_info += attr_count[attr_val] * self.compute_entropy(labels[ids])
            attr_gain = parent_info - attr_split_info
            attr_gain_ratio = self.compute_dict_entropy(attr_count) * attr_gain
           
            if best_gain_ratio < attr_gain_ratio:
                best_attr_index = i
                best_info_gain = attr_gain
                best_gain_ratio = attr_gain_ratio
                best_attr_keys = attr_count.keys()
        if best_gain_ratio <= 0 :
            parent.leaf = True
            return [] # TO Check    
        else:
            parent.attribute_index =  best_attr_index
            parent.children = { i: Node() for i in best_attr_keys}
            to_return = []
            if best_attr_index != 0:
                used_attr_index.append(best_attr_index)
                for i in best_attr_keys:
                    inds = np.where(data[:, best_attr_index] == i)[0]
                    to_return.append( (parent.children[i], data[inds], labels[inds], used_attr_index) )
            else:
                parent.sent_len_split_val = sent_len_split_val
                to_return.append( (parent.children[0], data[le_ids], labels[le_ids], used_attr_index) )
                to_return.append( (parent.children[1], data[gt_ids], labels[gt_ids], used_attr_index) )
            return to_return
       
    def build_tree(self, data, labels):
        traversal_q = queue.Queue()
        root = Node()
        traversal_q.put_nowait( (root, data, labels, [] ))
        while not traversal_q.empty():
            node_to_split = traversal_q.get_nowait()
            child_nodes = self.split_node(*node_to_split)
            for child in child_nodes:
                traversal_q.put_nowait(child)
        self.root = root
        return root
       
    def split_infer(self, node, data, data_indices):
        if node.leaf:
            return (True, data_indices, np.zeros( (data.shape[0]), dtype = np.int32) + node.majority_class)
        else:
            to_queue = []
            if(node.attribute_index == 0):
                left_idx = np.where(data[:,0] <= node.sent_len_split_val)[0]
                right_idx = np.where(data[:,0] > node.sent_len_split_val)[0]
                to_queue.append( (node.children[0], data[left_idx], data_indices[left_idx]) )
                to_queue.append( (node.children[1], data[right_idx], data_indices[right_idx]) )
                return (False, to_queue)
            else:
                for i in node.children.keys():
                    split_inds = np.where( data[:, node.attribute_index]  == i)[0]
                    if len(split_inds) > 0:
                        to_queue.append( (node.children[i], data[split_inds], data_indices[split_inds]) )
                return (False, to_queue)
   
    def get_labels(self, data):
        root = self.root
        data_idx = np.arange(data.shape[0], dtype = np.int32)
        labels = np.zeros( (data.shape[0]), dtype = np.int32) + -1
        traversal_q = queue.Queue()
        traversal_q.put_nowait( (root, data, data_idx ))
        while not traversal_q.empty():
            node_to_split = traversal_q.get_nowait()
            split_return = self.split_infer(*node_to_split)
            if split_return[0]:
                labels[split_return[1]] = split_return[2]
            else:
                for child in split_return[1]:
                    traversal_q.put_nowait(child)
        return labels

def get_scores(Y_test_idx, y_pred_test):
    acc = (y_pred_test == Y_test_idx).mean()
    prec, rec, fscore, _ = precision_recall_fscore_support(Y_test_idx, y_pred_test, average='weighted')
    return acc, prec, rec, fscore
