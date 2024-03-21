### Task 1

为了实现 `_block_to_squares` 函数，你需要遍历 `Block` 对象（即方块游戏中的一个方块），并将其转换为渲染器能够理解的形式。这里的关键是理解每个 `Block` 可以表示一个未被划分的单一方块或者是由四个子方块组成的更复杂的结构。当一个 `Block` 未被划分时，我们只需要将它自己的颜色、位置和大小作为一个元组加入到列表中；当一个 `Block` 被划分时，我们需要递归地对其子方块进行同样的处理。

以下是 `_block_to_squares` 函数的实现策略：

1. 检查当前的 `Block` 是否被划分（即是否有子方块）。
2. 如果没有被划分（`children` 列表为空），则直接使用当前 `Block` 的颜色、位置和大小创建一个元组，并将这个元组添加到列表中。
3. 如果被划分了（`children` 列表包含四个子方块），则对每个子方块递归调用 `_block_to_squares` 函数，并将结果合并到当前列表中。
4. 返回这个列表。

现在，让我们用代码实现这个策略：

```python
def _block_to_squares(block: Block) -> list[tuple[tuple[int, int, int], tuple[int, int], int]]:
    squares = []
    if not block.children:
        # 如果这个 Block 没有子方块，直接添加它自己的信息
        squares.append((block.colour, block.position, block.size))
    else:
        # 如果这个 Block 有子方块，递归地为每个子方块调用此函数，并将结果添加到 squares 列表中
        for child in block.children:
            squares.extend(_block_to_squares(child))
    return squares
```

在实现这个函数时，请确保你已经正确理解了 `Block` 类的属性，特别是 `children` 属性，以及如何根据这些属性来计算子方块的位置。`Block` 类提供了所有必要的信息来计算子方块的位置：当一个 `Block` 被划分时，它的四个子方块的大小是它自己的一半，位置根据父 `Block` 的位置和大小来确定。



### Task 2

要实现 `Block.smash` 方法，你需要遵循方法的文档字符串（docstring）中的描述，确保当一个方块（`Block`）成功进行“smash”操作时，它会生成四个子方块，并且自身的颜色属性变为 `None`。"smash"操作只有在方块没有子方块且其级别（`level`）小于最大深度（`max_depth`）时才能进行。

以下是实现 `Block.smash` 方法的步骤：

1. 检查当前方块是否没有子方块且其级别小于最大深度。
2. 如果以上条件满足，当前方块生成四个子方块。
3. 对于每个新生成的子方块：
    - 使用 `random.random` 生成一个 `[0, 1)` 区间内的随机数。
    - 如果这个随机数小于 `math.exp(-0.25 * child_level)`（其中 `child_level` 是子方块的级别），那么这个子方块也进行一次 "smash" 操作。
    - 如果子方块没有被 "smash"，则随机从 `settings.COLOUR_LIST` 中选择一个颜色赋值给它。
4. 设置当前方块的颜色为 `None`，表示它现在是由子方块组成的。
5. 返回 `True` 表示 "smash" 操作成功。

下面是实现这个方法的示例代码：

```python
def smash(self) -> bool:
    if self.children or self.level >= self.max_depth:
        return False  # 如果已经有子方块或者达到最大深度，则不进行 smash 操作

    self.colour = None  # 清除当前方块的颜色

    child_position = self.children_positions()

    for i in range(4):  # 为当前方块生成四个子方块
        child_size = self.child_size()
        child_colour = random.choice(COLOUR_LIST)  # 随机选择颜色
        child_level = self.level + 1
        child = Block(child_position[i], child_size, child_colour, child_level, self.max_depth)
        self.children.append(child)

        # 决定是否对子方块进行 smash 操作
        if random.random() < math.exp(-0.25 * child_level):
            child.smash()  # 递归调用 smash
        else:
            child.colour = child_colour  # 如果不进行 smash，则赋予随机颜色

    return True
```

请注意，`_calculate_child_position` 方法需要你根据当前方块的位置和大小来计算每个子方块的位置。这部分的具体实现取决于你的方块布局逻辑，特别是考虑到四个子方块分别位于当前方块的四个象限中。



### Task 3

为了实现 `generate_goals` 函数和 `BlobGoal` 与 `PerimeterGoal` 类中的 `description` 方法，我们首先需要理解这些组件的作用以及如何实现它们。`generate_goals` 函数用于生成一系列的目标（`Goal`），这些目标是随机选择的，并且每个目标都有一个不同的随机生成的颜色。`BlobGoal` 和 `PerimeterGoal` 是 `Goal` 的两个子类，分别代表两种不同的游戏目标。

#### 实现 `generate_goals` 函数

1. 首先，你需要从颜色列表（`COLOUR_LIST`）中随机选择不同的颜色，以确保没有两个目标具有相同的颜色。
2. 然后，随机选择目标类型（`BlobGoal` 或 `PerimeterGoal`），并为每个目标指定一个随机选择的颜色。
3. 返回这个目标列表。

####  实现`description`方法

对于 `description` 方法，你需要返回一个字符串描述，说明目标的类型以及它适用的颜色。使用 `colour_name` 函数来获取颜色的名称。

以下是可能的实现方式：

```python
import random
from settings import COLOUR_LIST

def generate_goals(num_goals: int) -> list[Goal]:
    chosen_colours = random.sample(COLOUR_LIST, num_goals)  # 随机选择不同的颜色
    goals = []
    for colour in chosen_colours:
        # 随机选择目标类型并创建目标
        goal_type = random.choice([BlobGoal, PerimeterGoal])
        goals.append(goal_type(colour))
    return goals

class PerimeterGoal(Goal):
    def description(self) -> str:
        # 使用 colour_name 函数获取颜色名称
        return f"Maximize the placement of {colour_name(self.colour)} blocks on the border. Corner colors will score double points."

class BlobGoal(Goal):
    def description(self) -> str:
        # 使用 colour_name 函数获取颜色名称
        return f"Create as large a {colour_name(self.colour)} colored connected area as possible."
```

注意：上面的代码示例假设 `COLOUR_LIST` 已经定义且包含了你想使用的颜色，同时 `colour_name` 函数能够根据颜色值返回相应的颜色名称。你可能需要根据你的具体实现细节对代码进行调整。此外，`colour_name` 函数的实现依赖于颜色与名称之间的映射，你需要确保这个映射是正确的，并且能够处理找不到颜色名称时抛出的 `UnknownColourError` 异常。

### Task 4

#### 实现`_get_block`函数

为了实现 `_get_block` 函数，你需要根据给定的 `Block`（方块），位置（`location`），和级别（`level`）来查找并返回在指定级别上且包含给定位置的 `Block`。如果在给定位置没有找到对应级别的 `Block`，则返回 `None`。这个函数对于实现玩家通过鼠标悬停在游戏板上来选择方块的功能至关重要。

以下是 `_get_block` 函数的大致实现逻辑：

1. 首先，检查给定的位置是否在根 `Block` 的范围内。如果不在，立即返回 `None`。
2. 如果当前 `Block` 在指定的 `level` 或者当前 `Block` 已经是最底层的 `Block`（即没有子 `Block`），则返回当前 `Block`。
3. 如果当前 `Block` 有子 `Block` 且未达到指定的 `level`，递归地检查哪个子 `Block` 包含给定的位置，并继续在该子 `Block` 上进行搜索。
4. 如果在递归过程中找到了符合条件的 `Block`，则返回它；否则，如果没有找到符合条件的 `Block`，则返回 `None`。

以下是根据上述逻辑的实现代码示例：

```python
def _get_block(block: Block, location: tuple[int, int], level: int) -> Block | None:
    # 检查给定的位置是否在当前 Block 的范围内
    x, y = location
    top_left_x, top_left_y = block.position
    bottom_right_x = top_left_x + block.size
    bottom_right_y = top_left_y + block.size

    # 检查位置是否在当前 Block 内部或顶部/左侧边缘
    if not (top_left_x <= x < bottom_right_x and top_left_y <= y < bottom_right_y):
        return None  # 如果位置不在当前 Block 内，则返回 None

    # 如果当前 Block 是所需级别或已经是最底层 Block，则返回当前 Block
    if block.level == level or not block.children:
        return block

    # 递归检查哪个子 Block 包含给定位置
    for child in block.children:
        result = _get_block(child, location, level)
        if result is not None:
            return result

    # 如果没有找到符合条件的子 Block（理论上不应该发生），返回 None
    return None
```

#### 实现`create_players`函数

为了实现 `create_players` 函数，你首先需要生成足够多的随机目标，以分配给每个玩家一个。这可以通过之前实现的 `generate_goals` 函数来完成。然后，根据函数参数创建相应数量和类型的玩家，并为每个玩家分配一个唯一的ID和一个随机目标。玩家ID从0开始，按照创建顺序递增。

下面是如何实现 `create_players` 函数的示例代码：

```python
def create_players(num_human: int, num_random: int, smart_players: list[int]) -> list[Player]:
    total_players = num_human + num_random + len(smart_players)
    goals = generate_goals(total_players)  # 生成足够多的随机目标
    players = []

    player_id = 0

    # 创建 HumanPlayer 对象
    for _ in range(num_human):
        players.append(HumanPlayer(player_id, goals.pop(0)))
        player_id += 1

    # 创建 RandomPlayer 对象
    for _ in range(num_random):
        # 假设 RandomPlayer 已经定义，且有类似于 HumanPlayer 的初始化方法
        players.append(RandomPlayer(player_id, goals.pop(0)))
        player_id += 1

    # 创建 SmartPlayer 对象
    for difficulty in smart_players:
        # 假设 SmartPlayer 已经定义，且可以接受一个难度级别作为参数
        players.append(SmartPlayer(player_id, goals.pop(0), difficulty))
        player_id += 1

    return players
```

在这个示例中，`HumanPlayer`、`RandomPlayer` 和 `SmartPlayer` 都是假定已经实现的玩家类型。每种类型的玩家都通过他们的初始化方法被赋予一个独一无二的ID和一个从 `goals` 列表中取出的随机目标

### Task 5

#### 实现 `_update_children_positions` 方法

需要根据父块的位置来递归更新所有子代块的位置，确保它们在视觉上正确地相对于父块排列。考虑到每个子块的位置取决于其在父块内的相对位置，这个方法需要逐一更新子块的位置，并对每个子块递归地调用相同的更新过程。

这里是一个基于 `Block` 类的实现示例：

```python
def _update_children_positions(self, position: tuple[int, int]) -> None:
    self.position = position
    if self.children:
        # 确定子块大小（假设所有子块大小相等）
        child_size = self.size // 2

        # 计算并更新四个子块的位置
        positions = [(position[0] + child_size, position[1]),  # 上右
                     (position[0], position[1]),  # 上左
                     (position[0], position[1] + child_size),  # 下左
                     (position[0] + child_size, position[1] + child_size)]  # 下右

        for i, child in enumerate(self.children):
            # 递归更新子块及其后代的位置
            child._update_children_positions(positions[i])
```

这个方法首先设置当前块的位置为给定的 `position`。如果当前块有子块（即它不是叶节点），它会计算每个子块的新位置。子块的位置根据父块的位置和大小进行计算，以确保子块在父块内正确排列。然后，这个方法递归地对每个子块调用 `_update_children_positions` 方法，以同样的方式更新这些子块的子代位置。

请注意，这个实现假定每个 `Block` 都是正方形，且子块的大小是父块大小的一半。这个方法确保了当改变一个 `Block` 的位置时，它所有的子代块的位置都会相应地更新，以保持整个块结构的一致性。

#### 实现`Rotate`方法

了解到需要通过改变子块（`children`）的位置（`position`）来实现旋转，而不仅仅是改变子块列表的顺序，下面将提供一个修正的版本来处理旋转。

旋转一个块（不仅仅是改变其子块的顺序）实际上涉及到两个关键步骤：

1. 根据旋转方向重新计算子块的位置。
2. 递归地旋转所有子块。

考虑到 `Block` 类定义中子块的位置是依据父块的位置和大小确定的，当执行旋转时，每个子块的新位置需要根据旋转方向来重新计算。这里假设父块的左上角是 `(px, py)`，大小为 `s`，子块的大小自然是 `s/2`。对于顺时针旋转：

- 原本位于上右的子块（0）移动到下右（3）的位置。
- 上左（1）的子块移动到上右（0）的位置。
- 下左（2）的子块移动到上左（1）的位置。
- 下右（3）的子块移动到下左（2）的位置。

以下是考虑位置更新的 `rotate` 方法实现：

```python
pythonCopy codeROT_CW = 1  # 顺时针旋转标识符
ROT_CCW = -1  # 逆时针旋转标识符

def rotate(self, direction: int) -> bool:
    if not self.children:  # 没有子块，无需旋转
        return False

    half_size = self.size // 2
    positions = [(self.position[0] + child_size, self.position[1]),
                      self.position,
                     (self.position[0], self.position[1] + child_size),
                     (self.position[0] + child_size, self.position[1] + child_size)]

    if direction == ROT_CW:
        # 顺时针旋转子块位置
        new_positions = [positions[3], positions[0], positions[1], positions[2]]
    elif direction == ROT_CCW:
        # 逆时针旋转子块位置
        new_positions = [positions[1], positions[2], positions[3], positions[0]]
    else:
        return False

    # 更新子块位置并递归旋转
    for i, child in enumerate(self.children):
        child.position = new_positions[i]
        child.rotate(direction)

    return True
```

这个实现首先计算了旋转前子块的位置，然后根据旋转方向确定新位置，并更新每个子块的 `position` 属性。之后，它递归地旋转每个子块。这种方法确保了旋转操作正确地反映在了整个块结构上，包括块的视觉呈现。

#### 实现`Swap`方法

这段代码通过递归方式处理了 `swap` 操作的核心问题，即确保不仅当前 `Block` 的子块位置更新，而且这些子块的所有后代（孙子块等）也根据交换后的新位置进行相应的位置更新，保持了整体的位置关系不变。这种方法有效地解决了子块及其子孙块位置更新的问题，让我们逐步分析：

`swap` 方法

- 首先，这个方法检查当前 `Block` 是否有子块。如果没有，直接返回 `False` 表示没有执行交换。
- 然后，它根据 `direction`（交换方向）计算了子块在水平或垂直交换后应该有的新位置。
- 然后，调用`_update_children_positions`更新当前block内子块的位置
- 最后，根据交换方向更新 `children` 列表中子块的顺序，以反映水平或垂直交换。

这个改进后的实现有效地处理了所有相关块的位置更新，确保了即使在进行了交换操作之后，整个块结构的视觉表示仍然是正确的。这种方法充分考虑了 `Block` 树结构的特点，通过递归确保了从顶层块到最底层块的整体一致性。

```python
def swap(self, direction: int) -> bool:
    """Swap the child Blocks of this Block.

    If this Block has no children, do nothing. Otherwise, if <direction> is
    SWAP_VERT, swap vertically.
    If <direction> is SWAP_HORZ, swap horizontally.

    Return True iff the swap was performed.

    Precondition:
    - <direction> is either (SWAP_VERT, SWAP_HORZ)
    """
    # TODO: Implement this method
    if not self.children:  # 如果没有子块，不进行交换
        return False

    child_size = self.child_size()
    positions = [(self.position[0] + child_size, self.position[1]),
                 self.position,
                 (self.position[0], self.position[1] + child_size),
                 (self.position[0] + child_size, self.position[1] + child_size)]

    if direction == SWAP_HORZ:
        # 水平交换: 计算新位置并更新子块位置
        # 交换上左(1)与上右(0)，下左(2)与下右(3)
        new_positions = [positions[1], positions[0], positions[3], positions[2]]
    elif direction == SWAP_VERT:
        # 垂直交换: 计算新位置并更新子块位置
        # 交换上右(0)与下右(3)，上左(1)与下左(2)
        new_positions = [positions[3], positions[2], positions[1], positions[0]]
    else:
        return False

    for i, child in enumerate(self.children):
        child._update_children_positions(new_positions[i])

    if direction == SWAP_HORZ:
        self.children = [self.children[1], self.children[0], self.children[3], self.children[2]]
    elif direction == SWAP_VERT:
        self.children = [self.children[3], self.children[2], self.children[1], self.children[0]]

    return True
```

#### 实现`Paint`方法

要实现 `paint` 方法，你需要检查当前 `Block` 是否满足两个条件：1) 它是一个在最大深度（`max_depth`）上的单元格（即它没有子块，可以理解为叶节点）；2) 它的当前颜色与要改变成的颜色不同。如果这两个条件都满足，那么你可以改变这个 `Block` 的颜色。

这里是如何实现这个方法的示例：

```python
def paint(self, colour: tuple[int, int, int]) -> bool:
    # 检查当前 Block 是否是最深的单元且颜色不同
    if self.level == self.max_depth and self.colour != colour:
        self.colour = colour  # 改变颜色
        return True  # 表示颜色已改变
    return False  # 如果不满足条件，返回 False
```

这段代码的逻辑非常直接。它首先检查当前 `Block` 是否为叶节点且位于最大深度（这是通过比较 `level` 和 `max_depth` 属性来确定的），同时还检查当前颜色是否与指定颜色不同。如果这些条件都满足，那么就更新 `Block` 的颜色，并返回 `True` 表示颜色已经成功更改。如果不满足这些条件，方法将返回 `False`。

#### 实现 `combine` 方法

需要首先确认当前 `Block` 是否已经被划分为子块，并且这些子块都是叶节点（即它们没有自己的子块）。然后，你需要计算子块中的多数颜色（如果存在的话），并根据这个多数颜色更新当前 `Block` 的颜色，同时去除所有子块，使得当前 `Block` 成为叶节点。

以下是实现这一方法的示例步骤：

1. 检查当前 `Block` 是否有四个子块，且这些子块都是叶节点。
2. 统计每种颜色的子块数量。
3. 确定是否存在多数颜色。
4. 如果存在多数颜色，更新当前 `Block` 的颜色，移除所有子块，并返回 `True`。
5. 如果不满足条件或没有多数颜色，不做任何操作并返回 `False`。

示例代码如下：

```python
pythonCopy codedef combine(self) -> bool:
    # 检查是否有四个子块且都是叶节点
    if len(self.children) != 4 or any(child.children for child in self.children):
        return False

    # 统计每种颜色的子块数量
    colour_counts = {}
    for child in self.children:
        if child.colour in colour_counts:
            colour_counts[child.colour] += 1
        else:
            colour_counts[child.colour] = 1

    # 寻找多数颜色
    max_count = max(colour_counts.values())
    # 检查是否存在多数颜色，且没有并列
    if list(colour_counts.values()).count(max_count) == 1:
        for colour, count in colour_counts.items():
            if count == max_count:
                # 更新颜色，移除子块
                self.colour = colour
                self.children = []
                return True

    return False
```

请注意，这段代码的执行依赖于 `Block` 类和其 `children` 属性的具体实现。它首先检查当前 `Block` 是否满足操作的前提条件，即它有四个子块且这些子块都没有自己的子块（即都是叶节点）。然后，它计算每种颜色的子块数量，并检查是否存在多数颜色。如果找到了多数颜色，它就更新当前 `Block` 的颜色，并移除所有子块，使当前 `Block` 成为叶节点。如果不满足条件或没有多数颜色，则不执行任何操作并返回 `False`。

### Task 6

#### 实现`flatten`函数

`flatten` 函数返回的格式是 `list[list[tuple[int, int, int]]]`，其中每个内部列表代表游戏板上的一行，每个元组代表该位置上的颜色（以RGB格式）。根据这个格式，下面是如何递归地实现这个函数，使其正确返回一个二维列表，以表示整个 `Block` 的颜色结构：

1. 基础情况：如果当前 `Block` 是叶节点（即它没有子块），那么创建一个大小为 `2^{max_depth - level}` 的二维列表，每个元素都设置为当前 `Block` 的颜色。
2. 递归情况：如果当前 `Block` 有子块，对每个子块调用 `flatten` 函数，然后将返回的二维列表组合成当前 `Block` 的完整二维列表。

这里是具体实现：

```python
def flatten(block: Block) -> list[list[tuple[int, int, int]]]:
    # 计算当前 block 需要生成的二维列表的大小
    size = 2 ** (block.max_depth - block.level)
    # 如果当前 block 是叶节点，返回一个填充了 block 颜色的二维列表
    if not block.children:
        return [[block.colour for _ in range(size)] for _ in range(size)]
    else:
        # 计算每个子块的二维列表，然后将它们组合在一起
        child_size = size // 2
        top_left = flatten(block.children[0])
        top_right = flatten(block.children[1])
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
```

这个实现中，`flatten` 函数首先检查当前 `Block` 是否是叶节点。如果是，它创建并返回一个只包含当前 `Block` 颜色的二维列表。如果当前 `Block` 有子块，它递归地对每个子块调用 `flatten`，然后将得到的四个二维列表（代表四个子块的颜色结构）组合成一个大的二维列表，这个大的二维列表就代表了整个 `Block` 的颜色结构。通过这种方式，无论 `Block` 结构多么复杂，`flatten` 都能生成一个准确的二维颜色表示。

#### 实现`PerimeterGoal.score`方法

为了实现 `PerimeterGoal` 类中的 `score` 方法，我们需要使用之前定义的 `flatten` 函数来获得游戏板的二维颜色表示。得到这个表示后，就可以遍历游戏板的边界（即周长）来计算目标颜色的单位格数。需要注意的是，角落的单元格需要被计算两次。

步骤如下：

1. 使用 `flatten` 函数得到游戏板的二维颜色表示。
2. 遍历这个二维列表的边界，包括顶部、底部、左侧和右侧。
3. 对于每个边界单元格，如果它的颜色与目标颜色相同，则增加得分。
4. 对于四个角落的单元格，如果颜色与目标颜色相同，则额外增加一分（因为角落单元格本身就需要计算两次）。

下面是具体实现：

```python
def score(self, board: Block) -> int:
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
```

这段代码首先获取了游戏板的二维颜色表示，然后分别遍历了边界上的单元格，检查每个单元格的颜色是否与目标颜色相匹配。对于边界上的每个匹配单元格，分数增加1；对于四个角落的单元格，如果颜色匹配，则由于它们在前面的步骤中已经被计算了一次，这里额外增加1分以满足“角落单元格计数两次”的规则。

### Task 7

#### 实现`BlobGoal._undiscovered_blob_size`方法

为了实现 `_undiscovered_blob_size` 方法，你需要使用递归来遍历板上的单元格，并计算特定颜色的最大连通块（blob）的大小。该方法将根据给定的起始位置 `pos` 和目标颜色，在已平展的游戏板 `board` 上寻找并计算blob的大小，同时利用 `visited` 列表来防止重复计算同一单元格。

以下是实现这个方法的步骤：

1. 首先，检查 `pos` 是否在 `board` 的边界内。如果不在，直接返回0。
2. 检查当前单元格是否已经访问过（通过 `visited` 列表）。如果已访问，返回0。
3. 如果当前单元格是目标颜色，将其标记为已访问（`visited` 对应位置设为1），否则标记为已访问但不是目标颜色（设为0），并返回0。
4. 对当前单元格的每个邻居（上、下、左、右），如果它是目标颜色且未访问，递归调用 `_undiscovered_blob_size` 来计算邻居单元格的blob大小，并将这些大小累加到当前单元格的blob大小中。

这里是方法的一个基本框架：

```python
def _undiscovered_blob_size(self, pos: tuple[int, int],
                            board: list[list[tuple[int, int, int]]],
                            visited: list[list[int]]) -> int:
    row, col = pos
    # 检查位置是否越界
    if row < 0 or row >= len(board) or col < 0 or col >= len(board[0]):
        return 0

    # 检查当前单元格是否已经访问
    if visited[row][col] != -1:
        return 0

    # 检查当前单元格颜色
    if board[row][col] != self.colour:
        visited[row][col] = 0
        return 0
    else:
        visited[row][col] = 1  # 标记为已访问且是目标颜色
        blob_size = 1  # 当前单元格至少为1

    # 遍历四个方向的邻居
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for d in directions:
        new_pos = (row + d[0], col + d[1])
        blob_size += self._undiscovered_blob_size(new_pos, board, visited)

    return blob_size
```

请确保 `self.colour` 是定义了目标颜色的属性。这段代码首先检查给定位置是否越界或已经被访问，然后根据当前单元格的颜色来决定接下来的操作。如果当前单元格是目标颜色，它会将自己计入blob大小，并递归地检查其邻居。这种递归方法能够有效地遍历并计算特定颜色的blob的大小，同时 `visited` 列表确保每个单元格只被计算一次。

#### 实现`BlobGoal.score`方法

要实现 `BlobGoal.score` 方法，你需要使用 `_undiscovered_blob_size` 作为辅助方法来找出游戏板上目标颜色的最大连通块（blob）的大小。这涉及到首先平展（flatten）游戏板，然后迭代平展后的每个单元格，使用 `_undiscovered_blob_size` 来计算如果该单元格属于目标颜色的 blob，则该 blob 的大小。通过这个过程，你可以找到并返回最大 blob 的大小作为当前目标的得分。

以下是如何实现 `score` 方法的步骤：

1. 使用 `flatten` 函数获取游戏板的二维颜色表示。
2. 初始化一个与二维颜色表示同维度的 `visited` 结构，用于标记访问过的单元格。
3. 遍历每个单元格，使用 `_undiscovered_blob_size` 计算 blob 的大小，并更新最大 blob 大小。
4. 返回最大 blob 大小作为得分。

这里是具体的实现代码：

```python
def score(self, board: Block) -> int:
    flattened_board = flatten(board)  # 获取平展后的游戏板
    max_blob_size = 0  # 初始化最大 blob 大小为 0
    visited = [[-1 for _ in range(len(flattened_board))] for _ in range(len(flattened_board))]  # 初始化访问结构

    # 遍历每个单元格
    for row in range(len(flattened_board)):
        for col in range(len(flattened_board)):
            if visited[row][col] == -1 and flattened_board[row][col] == self.colour:
                # 如果单元格未被访问且颜色匹配，则计算 blob 大小
                blob_size = self._undiscovered_blob_size((row, col), flattened_board, visited)
                if blob_size > max_blob_size:
                    max_blob_size = blob_size  # 更新最大 blob 大小

    return max_blob_size
```

在这段代码中，`flatten(board)` 用于获取游戏板的平展表示，然后初始化一个 `visited` 结构来跟踪哪些单元格已经被考虑过。接下来，通过遍历每个单元格，并且对于每个未访问且颜色匹配的单元格，调用 `_undiscovered_blob_size` 来计算该单元格所属 blob 的大小。最后，返回遍历过程中找到的最大 blob 大小作为得分。

请确保 `self.colour` 正确设置了目标颜色，以便于 `score` 方法可以正确地识别和计算目标颜色的 blob 大小。

### Task 8

#### 实现 `create_copy` 方法

为了实现 `Block.create_copy` 方法，我们需要创建当前 `Block` 的一个深拷贝，这意味着每一级的块都需要是新的块（不是别名），包括所有的子块。此外，为了正确比较两个 `Block` 对象是否相等，还提供了一个 `__eq__` 方法实现，用于判断两个 `Block` 对象及其所有后代是否完全相同。

创建 `Block` 的深拷贝需要递归地复制每个子块。对于没有子块的叶节点，直接创建一个新的 `Block` 实例即可。对于有子块的 `Block`，除了复制当前块的属性外，还需要对每个子块调用 `create_copy` 并将复制得到的子块添加到新块的子块列表中。

```python
def create_copy(self) -> Block:
    if not self.children:  # 如果当前 Block 是叶节点
        # 直接创建并返回一个新的 Block 实例
        return Block(self.position, self.size, self.colour, self.level, self.max_depth)
    else:
        # 创建一个新的 Block 实例，但暂时不包含子块
        new_block = Block(self.position, self.size, None, self.level, self.max_depth)
        # 递归复制每个子块并添加到新块的子块列表中
        new_block.children = [child.create_copy() for child in self.children]
        return new_block
```

#### 实现copy前后块的映射

根据提示，如果我们需要在 `board` 副本上找到一个有效的动作，并且之后要在原始 `board` 上应用这个动作，我们确实需要一种方法来找到副本中的 `Block` 在原始 `board` 中的对应项。利用 `Block` 对象的 `position` 和 `level` 属性可以实现这一点，因为这两个属性能唯一地标识 `board` 上的每个 `Block`，即使是在其副本中。

为了实现这个功能，你可以定义一个辅助函数，比如叫做 `_find_corresponding_block`，它接收原始 `board` 和副本中的 `Block` 的 `position` 和 `level`，返回原始 `board` 中对应的 `Block`。

这里是如何实现这个辅助函数的示例：

```python
def _find_corresponding_block(self, original_block: Block, position: tuple[int, int], level: int) -> Block | None:
    """
    在原始的 Block 树中找到与给定 position 和 level 对应的 Block。
    """
    # 如果当前块的 position 和 level 与给定值匹配，返回当前块
    if original_block.position == position and original_block.level == level:
        return original_block
    
    # 如果当前块有子块，递归搜索子块
    for child in original_block.children:
        result = self._find_corresponding_block(child, position, level)
        if result is not None:
            return result

    # 如果没有找到匹配的块，返回 None
    return None
```

#### 实现`RandomPlayer.generate_move` 方法

使其能够在不改变原始 `board` 的情况下生成一个有效的随机动作。关键是选择一个动作并尝试将其应用于 `board` 的副本上，以验证该动作是否有效。

为了简化实现，我们可以直接利用 `Action` 类及其子类提供的 `apply` 方法，这些方法已经为我们处理了如何在 `Block` 对象上执行特定的动作。我们的任务是在这些可用动作中随机选择一个，然后找到可以应用这个动作的 `Block` 对象。

以下是根据这些要求更新的 `RandomPlayer.generate_move` 方法实现：

```python
    def generate_move(self, board: Block) -> tuple[Action, Block] | None:
        # TODO: Implement this method
        if not self._proceed:
            # 如果不是玩家的回合，不生成移动
            return None

        possible_actions = list(KEY_ACTION.values())
        board_copy = board.create_copy()

        # 为了避免修改原始 board，尝试在 board 的副本上应用动作
        for _ in range(100):  # 限制尝试次数以避免无限循环
            action = random.choice(possible_actions)
            target_block = self._select_random_block(board_copy)

            extra_info = {}
            if action == PAINT:
                extra_info['colour'] = self.goal.colour  # 假设目标颜色存储在 goal 属性中

            success = action.apply(target_block, extra_info)
            if success:
                self._proceed = False  # 动作成功后等待下一次点击
                original_target_block = self._find_corresponding_block(board, target_block.position, target_block.level)
                return action, original_target_block  # 返回在原始 board 上的动作

        # 如果找不到有效动作，返回 None
        return None
```

这里有几个关键点：

1. 我们从 `KEY_ACTION.values()` 中随机选择动作。注意，我们不从 `KEY_ACTION.keys()` 中选择，因为我们关心的是动作本身，而不是触发这些动作的按键。
2. 对于每个选中的动作，我们在 `board` 的副本上尝试执行它。这样做是为了确保不会修改原始的 `board`。
3. 特别对于 `PAINT` 动作，我们需要额外的信息（如目标颜色），这通过 `extra_info` 字典传递给 `apply` 方法。
4. 如果动作成功应用，则返回动作和`original_target_block` 的组合。

### Task 9

在 `generate_move` 方法中，我们需要生成 `_num_test` 个随机但有效的移动，对每个移动评估其对玩家得分的潜在影响（考虑到动作的惩罚），然后选择得分最高的移动。如果没有找到任何能提高得分的移动，则选择 `PASS`。

```python
def generate_move(self, board: Block) -> tuple[Action, Block] | None:
    if not self._proceed:
        return None
    best_score = -float('inf')
    best_move = None
    current_score = self.goal.score(board)  # 当前板的得分，用于比较

    for _ in range(self._num_test):
        action = random.choice(list(KEY_ACTION.values()))  # 随机选择一个动作
        if action == PASS:  # 排除 PASS 动作
            continue

        board_copy = board.create_copy()
        target_block = self._select_random_block(board_copy)
        extra_info = {'colour': self.goal.colour} if action == PAINT else {}

        # 应用动作并评估结果
        if action.apply(target_block, extra_info):
            score_after_move = self.goal.score(board_copy) - action.penalty
            if score_after_move > best_score:
                best_score = score_after_move
                best_move = (action, target_block)

    # 如果找到的最佳移动比当前得分还要好，返回该移动
    if best_score > current_score:
        return best_move
    else:
        # 如果没有找到比当前得分更好的移动，执行 PASS
        return (PASS, board)
```

