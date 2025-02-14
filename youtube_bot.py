import gym  
import time  
import random  
from selenium import webdriver  
from selenium.webdriver.chrome.service import Service  
from selenium.webdriver.common.by import By  
from selenium.webdriver.common.keys import Keys  
from fake_useragent import UserAgent  
from stable_baselines3 import PPO  

YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=GCffvbfnnq0"  

# WebDriver Setup  
def setup_webdriver():  
    ua = UserAgent()  
    options = webdriver.ChromeOptions()  
    options.add_argument("--headless")  # Must run headless in GitHub Codespaces  
    options.add_argument("--no-sandbox")  
    options.add_argument("--disable-dev-shm-usage")  
    options.add_argument(f"user-agent={ua.random}")  

    service = Service("/usr/bin/chromedriver")  
    driver = webdriver.Chrome(service=service, options=options)  
    return driver  

# RL Training Environment  
class YouTubeEnv(gym.Env):  
    def __init__(self):  
        super(YouTubeEnv, self).__init__()  
        self.action_space = gym.spaces.Discrete(3)  
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=float)  

        self.driver = setup_webdriver()  
        self.driver.get(YOUTUBE_VIDEO)  
        time.sleep(5)  

    def step(self, action):  
        start_time = time.time()  

        if action == 0:  
            self.driver.execute_script("window.scrollBy(0, 500);")  
        elif action == 1:  
            self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.SPACE)  
        elif action == 2:  
            time.sleep(random.randint(5, 10))  

        watch_time = time.time() - start_time  
        reward = min(watch_time / 100, 1.0)  

        done = watch_time > random.randint(60, 120)  
        return [watch_time], reward, done, {}  

    def reset(self):  
        self.driver.get(YOUTUBE_VIDEO)  
        time.sleep(5)  
        return [0.5]  

    def close(self):  
        self.driver.quit()

# Train the RL Model  
env = YouTubeEnv()  
model = PPO("MlpPolicy", env, verbose=1)  
model.learn(total_timesteps=10000)  

# Save the trained model  
model.save("youtube_rl_model")
print("Model saved!")
