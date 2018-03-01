''' K-d Tree class
    Assume: all dimensions are  real number or integer
'''

import collections
import heapq

class KdTree:
    ''' K-d Tree class
    '''

    min_split = 1

    class KdTreeNode:
        ''' K-d Tree Node
        Fields:
            data: all data in the sub-tree, if the node is not a leaf node the field is None, else a list
            median: median of the data in the node
            split: demonstrates which dimension has been selected to split space
            left: left child node
            right: right child node
            parent: parent node
        '''

        data = None
        median = None
        left = None
        right = None
        parent = None
        split = -1

        
        def __init__(self, data=None, split=0, median=None, left=None, right=None, parent=None):
            ''' create KdTreeNode
            Args:
                data: all data in the sub-tree
                split: demonstrates which dimension has been selected to split space
                median: median in dimension 'split'
                left: left child node
                right: right child node
                parent: parent node
            '''
            self.data = data
            self.split = split
            self.median = median
            self.left = left
            self.right = right
            self.parent = parent

    def __init__(self, all_data, dimension, min_split = 1):
        ''' create KdTree
        Args:
            all_data: all data which forms the K-d tree
            dimension: how many dimensions the data has
            min_split: a leaf node contains no more than `min_split` data
        '''
        self.min_split = min_split
        self.dimension = dimension
        self.size = len(all_data)
        self.root = self.__build_kdtree(all_data)
        
    def __build_kdtree(self, all_data):
        if len(all_data) == 0:
            return None

        if len(all_data) <= self.min_split:
            return self.KdTreeNode(all_data, 0, all_data[0], None, None)
        
        split = self.__get_max_variance_dimension(all_data)
        left_data, median, right_data = self.__split(all_data, split)

        node = self.KdTreeNode([], split, median,
                self.__build_kdtree(left_data),
                self.__build_kdtree(right_data))
        if node.left is not None:
            node.left.parent = node
        if node.right is not None:
            node.right.parent = node
        return node



    def __get_max_variance_dimension(self, all_data):
        ''' get the dimension which has maximum variance
        Args:
            all_data: all data in the subtree
        '''
        num_of_data = len(all_data)

        means = [0.0] * self.dimension
        variances = [0.0] * self.dimension

        for data in all_data:
            for i in range(self.dimension):
                means[i] += data[i]

        means = [mean / num_of_data for mean in means]

        for data in all_data:
            for i in range(self.dimension):
                variances[i] += (data[i] - means[i]) ** 2
        
        return max((variance, i) for i, variance in enumerate(variances))[1]

    def __split(self, all_data, dim):
        ''' split data by median of dimension 'dim'
        Args:
            all_data: all data in the subtree
            dim: the dimension to split
        Returns:
            left_data: the data in left subtree
            median: the median data of dimension
            right_data: the data in right subtree
        '''
        length = len(all_data)
        
        def min_k(data, begin, end, k):
            # partition
            pivot = data[begin]
            left = begin
            right = end
            while left < right:
                while left < right and data[right][dim] >= pivot[dim]:
                    right -= 1
                data[left] = data[right]
                while left < right and data[left][dim] <= pivot[dim]:
                    left += 1
                data[right] = data[left]
            data[left] = pivot
            
            if left - begin + 1 == k:
                return
            elif left - begin + 1 < k:
                return min_k(data, left + 1, end, k - (left - begin + 1))
            else:
                return min_k(data, begin, left - 1, k)
        
        mid = round(length / 2)
        min_k(all_data, 0, length - 1, mid)
        return all_data[0:mid], all_data[mid - 1], all_data[mid:]

    def __repr__(self):
        preorder = []
        self.preorder(lambda x: preorder.append('({:.0f}, {:d})'.format(x.median[x.split], x.split) if x.left or x.right else str(x.data)))
        inorder = []
        self.inorder(lambda x: inorder.append('({:.0f}, {:d})'.format(x.median[x.split], x.split) if x.left or x.right else str(x.data)))

        ss = 'Min split: {:d}\n' \
             'Preorder: \n'      \
             '{:s}\n'            \
             'Inorder: \n'       \
             '{:s}\n'            \
             '{:s}'             # super __repr__
        
        result = ss.format(self.min_split, ' '.join(preorder), ' '.join(inorder), str(super().__repr__()))

        return result

    def preorder(self, visit):
        ''' preorder traverse
        Args:
            visit: visit function
        '''
        def helper(root):
            if root == None:
                return
            visit(root)
            helper(root.left)
            helper(root.right)
        
        helper(self.root)

    def inorder(self, visit):
        ''' inoder traverse
        Args:
            visit: visit function
        '''
        def helper(root):
            if root == None:
                return
            helper(root.left)
            visit(root)
            helper(root.right)
        
        helper(self.root)

    def data_distance(self, data_1, data_2):
        dis = 0.0
        for i in range(self.dimension):
            dis += (data_1[i] - data_2[i]) ** 2
        return dis ** (1 / 2)

    def search_k_nearest(self, data, k=1):        
        result = []
        min_radius = 999999.9

        if self.root is None:
            return result

        def helper(root):
            nonlocal result
            nonlocal min_radius
            if self.size == 0:
                return
            # leaf node
            if root.left is None and root.right is None:
                for src_data in root.data:
                    dis = self.data_distance(data, src_data)
                    if dis <= min_radius:
                        if len(result) == k:
                            heapq.heappop(result)
                        heapq.heappush(result, (-dis, src_data))
                        if len(result) == k:
                            min_radius = -result[0][0] # update threshold
            else:
                if data[root.split] <= root.median[root.split]:
                    helper(root.left)
                    if data[root.split] - root.median[root.split] >= -min_radius:
                        helper(root.right) # cycle cross rectangle
                else:
                    helper(root.right)
                    if data[root.split] - root.median[root.split] <= min_radius:
                        helper(root.left)
        
        helper(self.root)
        result.sort(key=(lambda x: -x[0]))
        result = [(data, -dis) for dis, data in result]
        return result

    def search_nearest(self, data):
        result = self.search_k_nearest(data, 1)
        return result[0] if len(result) == 1 else None
            

def test():
    x = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    # x = [1]
    t = KdTree(x, 2,)
    print(t)
    print(t.search_nearest([4, 3]))
    print(t.search_k_nearest([4, 3], 6))

if __name__ == '__main__':
    test()


