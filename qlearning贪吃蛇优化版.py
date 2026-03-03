import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import pickle
import json
import time

# 初始化Pygame
pygame.init()

# 游戏参数配置（缩小游戏区域，进一步减少状态空间）
WIDTH, HEIGHT = 400, 300  # 从800x600缩小到400x300，减少坐标状态数量
BLOCK_SIZE = 20            # 方块尺寸不变，保证视觉清晰
FPS = 30                   # 提高帧率，加快训练速度

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)          # 食物颜色
GREEN = (0, 255, 0)        # 蛇身颜色
BLUE = (0, 0, 255)         # 蛇头颜色
GRAY = (128, 128, 128)     # 边界颜色
YELLOW = (255, 255, 0)     # 按钮高亮
ORANGE = (255, 165, 0)     # 按钮颜色
PURPLE = (128, 0, 128)     # 标题颜色

# 游戏状态
GAME_START = 0
GAME_PLAYING = 1
GAME_OVER = 2

# Q-learning参数（优化后，加快收敛）
ALPHA = 0.2    # 提高学习率，加快Q值更新
GAMMA = 0.95   # 提高折扣因子，更关注长期奖励
EPSILON = 0.9  # 初始探索率
EPS_DECAY = 0.999  # 加快探索率衰减（从0.995→0.999）
MIN_EPS = 0.05     # 更低的最小探索率，后期几乎不探索

# 动作定义：0-上，1-右，2-下，3-左
ACTIONS = [0, 1, 2, 3]

class SnakeGame:
    def __init__(self):
        # 创建游戏窗口
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Q-Learning Snake")
        self.clock = pygame.time.Clock()
        self.game_state = GAME_START
        self.reset()

    def reset(self):
        """重置游戏状态（新一局游戏）"""
        # 初始化蛇的位置：中心位置，长度3
        self.snake = deque()
        start_x = WIDTH // 2 // BLOCK_SIZE * BLOCK_SIZE
        start_y = HEIGHT // 2 // BLOCK_SIZE * BLOCK_SIZE
        self.snake.append((start_x, start_y))
        self.snake.append((start_x - BLOCK_SIZE, start_y))
        self.snake.append((start_x - 2*BLOCK_SIZE, start_y))
        
        # 初始方向：右
        self.direction = 1
        
        # 生成食物
        self.food = self._generate_food()
        
        # 游戏状态
        self.game_over = False
        self.score = 0
        self.total_reward = 0  # 累计奖励
        self.steps = 0         # 步数
        
        return self._get_state()

    def _generate_food(self):
        """生成随机食物（避免落在蛇身上）"""
        while True:
            x = random.randint(1, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE - 1) * BLOCK_SIZE
            y = random.randint(1, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE - 1) * BLOCK_SIZE
            if (x, y) not in self.snake:
                return (x, y)

    def _get_state(self):
        """【核心修改】缩小状态空间：从9维→5维，仅保留关键决策信息
        新状态定义（5维离散特征）：
        (
            食物相对方向（上下左右）,  # 0-上，1-右，2-下，3-左
            危险上,                  # 1=危险，0=安全
            危险右,
            危险下,
            危险左
        )
        完全剔除坐标信息，仅保留“相对位置+危险”，状态空间缩小90%以上
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # 1. 食物相对方向（核心决策信息）
        if food_y < head_y:
            food_dir = 0  # 食物在上方
        elif food_x > head_x:
            food_dir = 1  # 食物在右方
        elif food_y > head_y:
            food_dir = 2  # 食物在下方
        else:
            food_dir = 3  # 食物在左方
        
        # 2. 危险检测（蛇头四方向是否有墙壁/蛇身）
        danger_up = 1 if (head_x, head_y - BLOCK_SIZE) in self.snake or head_y - BLOCK_SIZE < BLOCK_SIZE else 0
        danger_right = 1 if (head_x + BLOCK_SIZE, head_y) in self.snake or head_x + BLOCK_SIZE >= WIDTH - BLOCK_SIZE else 0
        danger_down = 1 if (head_x, head_y + BLOCK_SIZE) in self.snake or head_y + BLOCK_SIZE >= HEIGHT - BLOCK_SIZE else 0
        danger_left = 1 if (head_x - BLOCK_SIZE, head_y) in self.snake or head_x - BLOCK_SIZE < BLOCK_SIZE else 0
        
        # 最终状态（5维）
        state = (food_dir, danger_up, danger_right, danger_down, danger_left)
        return state

    def _take_action(self, action):
        """执行动作（改变方向），避免反向移动"""
        if action == 0 and self.direction != 2:    # 上
            self.direction = 0
        elif action == 1 and self.direction != 3:  # 右
            self.direction = 1
        elif action == 2 and self.direction != 0:  # 下
            self.direction = 2
        elif action == 3 and self.direction != 1:  # 左
            self.direction = 3

    def step(self, action):
        """【核心修改】强化奖励函数，增加方向奖励"""
        self.steps += 1
        reward = 0
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # 记录动作执行前与食物的距离
        prev_dist = abs(head_x - food_x) + abs(head_y - food_y)
        
        # 1. 执行动作
        self._take_action(action)
        
        # 2. 移动蛇头
        if self.direction == 0:    # 上
            new_head = (head_x, head_y - BLOCK_SIZE)
        elif self.direction == 1:  # 右
            new_head = (head_x + BLOCK_SIZE, head_y)
        elif self.direction == 2:  # 下
            new_head = (head_x, head_y + BLOCK_SIZE)
        else:                      # 左
            new_head = (head_x - BLOCK_SIZE, head_y)
        
        # 3. 碰撞检测（严厉惩罚）
        if (new_head[0] < BLOCK_SIZE or new_head[0] >= WIDTH - BLOCK_SIZE or
            new_head[1] < BLOCK_SIZE or new_head[1] >= HEIGHT - BLOCK_SIZE or
            new_head in self.snake):
            self.game_over = True
            reward = -50  # 加大撞墙惩罚（从-20→-50）
            self.total_reward += reward
            return self._get_state(), reward, self.game_over, self.score
        
        # 4. 添加新蛇头
        self.snake.appendleft(new_head)
        
        # 5. 检测是否吃到食物（高额奖励）
        if new_head == self.food:
            self.score += 1
            reward = 100  # 加大吃食物奖励（从10→100）
            self.food = self._generate_food()  # 重新生成食物
        else:
            # 没吃到食物：根据距离变化奖励/惩罚
            new_dist = abs(new_head[0] - food_x) + abs(new_head[1] - food_y)
            if new_dist < prev_dist:
                reward = 1  # 向食物移动，奖励
            elif new_dist > prev_dist:
                reward = -1  # 远离食物，惩罚
            else:
                reward = 0  # 距离不变，无奖励
            self.snake.pop()
        
        # 6. 步数惩罚（放宽阈值，从100×蛇长→200×蛇长）
        if self.steps > 200 * len(self.snake):
            reward = -10
            self.game_over = True
        
        self.total_reward += reward
        return self._get_state(), reward, self.game_over, self.score

    def draw_start_screen(self):
        """绘制开始游戏界面"""
        self.screen.fill(BLACK)
        
        # 标题
        font_large = pygame.font.SysFont('Arial', 40, bold=True)
        title_text = font_large.render("Q-Learning Snake", True, PURPLE)
        title_rect = title_text.get_rect(center=(WIDTH//2, HEIGHT//3))
        self.screen.blit(title_text, title_rect)
        
        # 副标题
        font_medium = pygame.font.SysFont('Arial', 20)
        subtitle_text = font_medium.render("Press SPACE to start", True, WHITE)
        subtitle_rect = subtitle_text.get_rect(center=(WIDTH//2, HEIGHT//2))
        self.screen.blit(subtitle_text, subtitle_rect)
        
        # 提示信息
        info_text = font_medium.render("Trained for 10000 episodes", True, GRAY)
        info_rect = info_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 40))
        self.screen.blit(info_text, info_rect)
        
        # 绘制蛇的预览
        for i in range(3):
            x = WIDTH//2 - BLOCK_SIZE * 2 + i * BLOCK_SIZE
            y = HEIGHT//3 + 60
            color = BLUE if i == 2 else GREEN
            pygame.draw.rect(self.screen, color, (x, y, BLOCK_SIZE, BLOCK_SIZE))
        
        # 绘制食物预览
        food_x = WIDTH//2 + BLOCK_SIZE * 2
        food_y = HEIGHT//3 + 60
        pygame.draw.rect(self.screen, RED, (food_x, food_y, BLOCK_SIZE, BLOCK_SIZE))
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def draw_game_over_screen(self):
        """绘制游戏结束界面"""
        self.screen.fill(BLACK)
        
        # 标题
        font_large = pygame.font.SysFont('Arial', 40, bold=True)
        game_over_text = font_large.render("Game Over", True, RED)
        game_over_rect = game_over_text.get_rect(center=(WIDTH//2, HEIGHT//3))
        self.screen.blit(game_over_text, game_over_rect)
        
        # 分数
        font_medium = pygame.font.SysFont('Arial', 24)
        score_text = font_medium.render(f"Final Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(WIDTH//2, HEIGHT//2))
        self.screen.blit(score_text, score_rect)
        
        # 奖励
        reward_text = font_medium.render(f"Total Reward: {self.total_reward:.1f}", True, WHITE)
        reward_rect = reward_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 40))
        self.screen.blit(reward_text, reward_rect)
        
        # 提示信息
        hint_text = font_medium.render("Press SPACE to restart", True, YELLOW)
        hint_rect = hint_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 80))
        self.screen.blit(hint_text, hint_rect)
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def render(self):
        """绘制游戏界面"""
        if self.game_state == GAME_START:
            self.draw_start_screen()
        elif self.game_state == GAME_OVER:
            self.draw_game_over_screen()
        else:  # GAME_PLAYING
            self.screen.fill(BLACK)
            
            # 绘制边界
            pygame.draw.rect(self.screen, GRAY, (0, 0, WIDTH, BLOCK_SIZE))
            pygame.draw.rect(self.screen, GRAY, (0, HEIGHT - BLOCK_SIZE, WIDTH, BLOCK_SIZE))
            pygame.draw.rect(self.screen, GRAY, (0, 0, BLOCK_SIZE, HEIGHT))
            pygame.draw.rect(self.screen, GRAY, (WIDTH - BLOCK_SIZE, 0, BLOCK_SIZE, HEIGHT))
            
            # 绘制食物
            pygame.draw.rect(self.screen, RED, (self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
            
            # 绘制蛇
            for i, (x, y) in enumerate(self.snake):
                color = BLUE if i == 0 else GREEN
                pygame.draw.rect(self.screen, color, (x, y, BLOCK_SIZE, BLOCK_SIZE))
            
            # 绘制分数和奖励
            font = pygame.font.SysFont(None, 24)  # 缩小字体，适配小窗口
            score_text = font.render(f"Score: {self.score}", True, WHITE)
            reward_text = font.render(f"Reward: {self.total_reward:.1f}", True, WHITE)
            self.screen.blit(score_text, (10, 10))
            self.screen.blit(reward_text, (10, 40))
            
            pygame.display.flip()
            self.clock.tick(FPS)

class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.epsilon = EPSILON

    def get_q_value(self, state, action):
        """获取状态-动作对的Q值（不存在则初始化为0）"""
        return self.q_table.get(state, {}).get(action, 0.0)

    def choose_action(self, state):
        """ε-greedy策略选择动作"""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(ACTIONS)
        q_values = [self.get_q_value(state, a) for a in ACTIONS]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        """更新Q表"""
        if done:
            target = reward
        else:
            next_q_values = [self.get_q_value(next_state, a) for a in ACTIONS]
            target = reward + GAMMA * max(next_q_values)
        
        current_q = self.get_q_value(state, action)
        new_q = current_q + ALPHA * (target - current_q)
        
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q
        
        # 加快探索率衰减
        if self.epsilon > MIN_EPS:
            self.epsilon *= EPS_DECAY

def train_agent(episodes=10000):
    """训练Q-learning智能体（10000轮）"""
    game = SnakeGame()
    agent = QLearningAgent()
    
    scores = []
    rewards = []
    avg_scores = []
    avg_rewards = []
    training_data = []
    
    start_time = time.time()
    
    for episode in range(episodes):
        state = game.reset()
        done = False
        total_reward = 0
        episode_start = time.time()
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            action = agent.choose_action(state)
            next_state, reward, done, score = game.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # 训练时不渲染，加快训练速度
        
        episode_time = time.time() - episode_start
        scores.append(score)
        rewards.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_reward = np.mean(rewards[-100:])
            avg_scores.append(avg_score)
            avg_rewards.append(avg_reward)
            max_score = np.max(scores[-100:])
            min_score = np.min(scores[-100:])
            elapsed_time = time.time() - start_time
            
            print(f"Episode {episode+1}/{episodes} | Avg Score: {avg_score:.2f} | Max Score: {max_score} | Min Score: {min_score} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f} | Time: {elapsed_time:.1f}s")
            
            training_data.append({
                'episode': episode + 1,
                'avg_score': float(avg_score),
                'max_score': int(max_score),
                'min_score': int(min_score),
                'avg_reward': float(avg_reward),
                'epsilon': float(agent.epsilon),
                'elapsed_time': float(elapsed_time)
            })
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("训练完成！详细统计信息：")
    print("="*60)
    print(f"总训练轮数: {episodes}")
    print(f"总训练时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"最终平均分数 (最后100轮): {np.mean(scores[-100:]):.2f}")
    print(f"最高分数: {np.max(scores)}")
    print(f"平均奖励 (最后100轮): {np.mean(rewards[-100:]):.2f}")
    print(f"Q表大小 (状态数): {len(agent.q_table)}")
    print(f"最终Epsilon: {agent.epsilon:.4f}")
    print("="*60)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(avg_scores)
    plt.title("Avg Score per 100 Episodes")
    plt.xlabel("100 Episodes")
    plt.ylabel("Avg Score")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(avg_rewards)
    plt.title("Avg Reward per 100 Episodes")
    plt.xlabel("100 Episodes")
    plt.ylabel("Avg Reward")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_results_10000.png", dpi=150)
    plt.close()
    print("\n训练曲线图已保存: training_results_10000.png")
    
    with open("training_data.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    print("训练数据已保存: training_data.json")
    
    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)
    print("Q表模型已保存: q_table.pkl")
    
    with open("trained_agent.pkl", "wb") as f:
        pickle.dump(agent, f)
    print("完整智能体已保存: trained_agent.pkl")
    
    return agent

def demo_agent(agent):
    """演示训练好的智能体"""
    game = SnakeGame()
    agent.epsilon = 0.0  # 关闭探索，只使用最优策略
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if game.game_state == GAME_START:
                        game.game_state = GAME_PLAYING
                        state = game.reset()
                        done = False
                    elif game.game_state == GAME_OVER:
                        game.game_state = GAME_PLAYING
                        state = game.reset()
                        done = False
        
        if game.game_state == GAME_PLAYING and not done:
            action = agent.choose_action(state)
            next_state, _, done, _ = game.step(action)
            state = next_state
            if done:
                game.game_state = GAME_OVER
        
        game.render()

if __name__ == "__main__":
    # 训练10000轮
    trained_agent = train_agent(episodes=10000)
    
    # 演示训练结果
    print("\n演示训练后的智能体...")
    demo_agent(trained_agent)
    
    pygame.quit()
