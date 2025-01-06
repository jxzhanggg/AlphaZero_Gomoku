# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)

@author: Junxiao Song
"""

import numpy as np
import copy
from operator import itemgetter

# 用来进行模拟的一个步骤，用这个策略直接把棋局下完，得到反馈的分数。
def rollout_policy_fn(board):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)

# 这个是一个返回一个可行动作列表，以及每个动作列表平均进行概率采样的策略函数。
def policy_value_fn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0

# TreeNode类并不记录局面，而是仅记录一个待价值“Q”，父节点，子节点的树。注意子节点列表的key是动作。value是TreeNode.
# 也即是说，TreeNode虽然实际对应着一个状态，但是并不在这个类中实际记录整个棋局状态，而是记录一连串的动作。
class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0 #访问的次数
        self._Q = 0
        self._u = 0
        self._P = prior_p

    # expand的输入是一个<行动，概率>对的列表，遍历整个列表，把所有没有选择的下一步行动进行扩展，全部放到当前节点的子节点中。
    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
    
    # select是选择子节点价值最大的那个，返回<对应动作, 子节点>
    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        # 这里等于是一个平均值，self._Q = sum(leaf_value_list) / len(leaf_value_list) （类似于胜率）
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            # 对于父节点，那么可知道执棋一方是相反的，所以对他得分是负的leaf_value.
            self._parent.update_recursive(-leaf_value)

        # 这里更新当前的价值得分
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        # 这里的公式大约等于价值+ 一个平衡性的价值，防止每次都选价值大的节点。
        # 这个公式有论文证明。
        # https://zhuanlan.zhihu.com/p/53948964 这里也有，应该alphazero里面也有，p.s. 我没看
        
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        # 搜索次数限制
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        # 从根节点出发
        node = self._root

        # 不断根据value，选择，并深入
        while(1):
            # 如果没有孩子节点，说明当前节点还没有有expand过，同时也说明没有update（只有经过了update，访问次数才会+1）
            # 如果是叶子节点了，那就可以选择action
            if node.is_leaf():
                
                break
            # Greedily select next move.
            # 如果当前节点访问过了，那么选择一个得分最高的
            # 注意，这时，node指向了选择后的节点。
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # 最终的node一定是叶子节点。
        # 看看有哪些地方可以下棋
        action_probs, _ = self._policy(state)
        # Check for end of game
        # 看看游戏结束了没
        end, winner = state.game_end()
        if not end:
            # 没结束就扩展当前节点，当前节点就产生了子节点。
            node.expand(action_probs)
        
        # 从当前节点出发进行一次模拟
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        # 对于当前棋局状态，当前可以下棋的人“player”如果获胜了，那么当前的状态得分减一分。这是为什么呢？
        # 因为造成这个局面的棋手是另一方，也就是说，另一方下了一个臭棋，导致“player”获胜了，那么对另一方其实是一个惩罚。
        # 对那个人来说，他的失误导致出现了这个局面。因此，他最好不要这样走棋。所以当前分数降低，下次就不会容易选择导致这个局面的action了。
        # 所以当前选择，永远是基于当前可以下棋的那个人最优的决策角度考虑的。他下棋走哪一步，要看下完后，生成的局面（也就是子节点）的分值怎么样。
        # 我们要能产生选高分的动作，也就是选择高分的子节点进行扩展。
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        # 模拟很多次，每一次会扩展一次，把一个叶子节点的可选择的行动，以及对应的子节点都挂上去。
        # 每一次会从相同的根出发，直到叶子节点，如果游戏没结束，则进行一次模拟，并在经历的路径上产生一次得分的全部更新。
        
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
            
        # 选择访问次数最多的，其实也就对应于价值最大的子节点。
        # 疑问：为什么不直接用价值最大？
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        # 如果这个动作在当前根节点的可选范围内，则移动到子节点，然后把子节点置为根。
        # 这个时候，子树仍然保存
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        # 如果不是，则直接生成一个新的根节点。这个时候，子树也没有了，完全重新初始化一个树。
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            # 模拟很多次，然后得到当前的行动
            move = self.mcts.get_move(board)
            # 行动完了，把树直接丢了，下一次从当前状态重头模拟？MCTS清空。
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
