from controller import ControllerEnv
import cv2

if __name__ == "__main__":
    # Load game over template image
    template = cv2.imread("./img/game_over.png", 0)  # 0: To grayscale
    assert template is not None, "Game Over image file not found!"
    # template = resize_cv2(template)
    env_config = {'mode': 'train',
                   'reload_template': template}



    env = ControllerEnv(env_config)

    # 0: black
    # 1: white
    obs = env.reset()


    while True:

        action = 1 if obs[50] < 0.6 else 0 # < Simple policy, and the entire script does not seem to work because of delay between the observation at the action
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

