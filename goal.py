"""CSC148 Assignment 2

CSC148 Winter 2024
Department of Computer Science,
University of Toronto

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

Authors: Diane Horton, David Liu, Mario Badr, Sophia Huynh, Misha Schwartz,
Jaisie Sin, and Joonho Kim

All of the files in this directory and all subdirectories are:
Copyright (c) Diane Horton, David Liu, Mario Badr, Sophia Huynh,
Misha Schwartz, Jaisie Sin, and Joonho Kim

Module Description:

This file contains the hierarchy of Goal classes and related helper functions.
"""
from __future__ import annotations
import math
import random
from block import Block
from settings import colour_name, COLOUR_LIST


def generate_goals(num_goals: int) -> list[Goal]:
    """Return a randomly generated list of goals with length <num_goals>.

    Each goal must be randomly selected from the two types of Goals provided
    and must have a different randomly generated colour from COLOUR_LIST.
    No two goals can have the same colour.

    Preconditions:
    - num_goals <= len(COLOUR_LIST)
    """
    # TODO: Implement this function
    chosen_colours = random.sample(COLOUR_LIST, num_goals)  # 随机选择不同的颜色
    goals = []
    for colour in chosen_colours:
        # 随机选择目标类型并创建目标
        goal_type = random.choice([BlobGoal, PerimeterGoal])
        goals.append(goal_type(colour))
    return goals


def flatten(block: Block) -> list[list[tuple[int, int, int]]]:
    """Return a two-dimensional list representing <block> as rows and columns of
    unit cells.

    Return a list of lists L, where,
    for 0 <= i, j < 2^{max_depth - self.level}
        - L[i] represents column i and
        - L[i][j] represents the unit cell at column i and row j.

    Each unit cell is represented by a tuple of 3 ints, which is the colour
    of the block at the cell location[i][j].

    L[0][0] represents the unit cell in the upper left corner of the Block.
    """
    # TODO: Implement this function
    # 计算当前 block 需要生成的二维列表的大小
    size = 2 ** (block.max_depth - block.level)
    # 如果当前 block 是叶节点，返回一个填充了 block 颜色的二维列表
    if not block.children:
        return [[block.colour for _ in range(size)] for _ in range(size)]
    else:
        # 计算每个子块的二维列表，然后将它们组合在一起
        child_size = size // 2
        top_left = flatten(block.children[1])
        top_right = flatten(block.children[0])
        bottom_left = flatten(block.children[2])
        bottom_right = flatten(block.children[3])

        # 创建一个新的二维列表来存储组合后的结果
        new_grid = []

        # 合并上半部分
        for i in range(child_size):
            new_grid.append(top_left[i] + top_right[i])
        # 合并下半部分
        for i in range(child_size):
            new_grid.append(bottom_left[i] + bottom_right[i])

        return new_grid


class Goal:
    """A player goal in the game of Blocky.

    This is an abstract class. Only child classes should be instantiated.

    Instance Attributes:
    - colour: The target colour for this goal, that is the colour to which
              this goal applies.
    """
    colour: tuple[int, int, int]

    def __init__(self, target_colour: tuple[int, int, int]) -> None:
        """Initialize this goal to have the given <target_colour>.
        """
        self.colour = target_colour

    def score(self, board: Block) -> int:
        """Return the current score for this goal on the given <board>.

        The score is always greater than or equal to 0.
        """
        raise NotImplementedError

    def description(self) -> str:
        """Return a description of this goal.
        """
        raise NotImplementedError


class PerimeterGoal(Goal):
    """A goal to maximize the presence of this goal's target colour
    on the board's perimeter.
    """

    def score(self, board: Block) -> int:
        """Return the current score for this goal on the given board.

        The score is always greater than or equal to 0.

        The score for a PerimeterGoal is defined to be the number of unit cells
        on the perimeter whose colour is this goal's target colour. Corner cells
        count twice toward the score.
        """
        # TODO: Implement this method
        flattened_board = flatten(board)  # 获取二维颜色表示
        score = 0
        size = len(flattened_board)

        # 遍历顶部和底部
        for i in range(size):
            if flattened_board[0][i] == self.colour:
                score += 1
            if flattened_board[size - 1][i] == self.colour:
                score += 1

        # 遍历左侧和右侧（不包括角落）
        for i in range(1, size - 1):
            if flattened_board[i][0] == self.colour:
                score += 1
            if flattened_board[i][size - 1] == self.colour:
                score += 1

        # 角落已经在顶部和底部计数，如果颜色匹配，则这里额外加一
        if flattened_board[0][0] == self.colour:  # 左上角
            score += 1
        if flattened_board[0][size - 1] == self.colour:  # 右上角
            score += 1
        if flattened_board[size - 1][0] == self.colour:  # 左下角
            score += 1
        if flattened_board[size - 1][size - 1] == self.colour:  # 右下角
            score += 1

        return score

    def description(self) -> str:
        """Return a description of this goal.
        """
        # TODO: Implement this method
        return f"Maximize the placement of {colour_name(self.colour)} blocks on the border. Corner colors will score double points."


class BlobGoal(Goal):
    """A goal to create the largest connected blob of this goal's target
    colour, anywhere within the Block.
    """

    def score(self, board: Block) -> int:
        """Return the current score for this goal on the given board.

        The score is always greater than or equal to 0.

        The score for a BlobGoal is defined to be the total number of
        unit cells in the largest connected blob within this Block.
        """
        # TODO: Implement this method
        return 148  # FIXME

    def _undiscovered_blob_size(self, pos: tuple[int, int],
                                board: list[list[tuple[int, int, int]]],
                                visited: list[list[int]]) -> int:
        """Return the size of the largest connected blob in <board> that (a) is 
        of this Goal's target <colour>, (b) includes the cell at <pos>, and (c)
        involves only cells that are not in <visited>.

        <board> is the flattened board on which to search for the blob.
        <visited> is a parallel structure (to <board>) that, in each cell,
        contains:
            -1 if this cell has never been visited
            0  if this cell has been visited and discovered
               not to be of the target colour
            1  if this cell has been visited and discovered
               to be of the target colour

        Update <visited> so that all cells that are visited are marked with
        either 0 or 1.

        If <pos> is out of bounds for <board>, return 0.
        """
        # TODO: Implement this method

    def description(self) -> str:
        """Return a description of this goal.
        """
        # TODO: Implement this method
        return f"Create as large a {colour_name(self.colour)} colored connected area as possible."


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'allowed-import-modules': [
            'doctest', 'python_ta', 'random', 'typing', 'block', 'settings',
            'math', '__future__'
        ],
        'max-attributes': 15
    })
