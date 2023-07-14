import time

import openai
import openai.error
import requests

from common.log import logger
from common.token_bucket import TokenBucket
from config import conf


# OPENAI提供的画图接口
class OpenAIImage(object):
    def __init__(self):
        openai.api_key = conf().get("open_ai_api_key")
        if conf().get("rate_limit_dalle"):
            self.tb4dalle = TokenBucket(conf().get("rate_limit_dalle", 50))

    def create_img(self, query, retry_count=0, api_key=None):
        try:
            if conf().get("rate_limit_dalle") and not self.tb4dalle.get_token():
                return False, "请求太快了，请休息一下再问我吧"
            logger.info("[OPEN_AI] image_query={}".format(query))
           
            base_url = conf().get("api_base_url")
            data = {
                'email': conf().get("api_email"), 
                'password': conf().get("api_password"),
            }
            r0 = requests.post(base_url + "/get_token", json=data).json()
            token_str = r0['info']
            model_name = 'Artist v0.3.0 Beta'
            neg_prompt = ""
            n_images=1
            scale=7
            output_size="960x960"
            select_seed=-1
            init_img=""
            controlnet_model=""

            data = {
                "token": token_str,
                "model_name": model_name,
                "prompt": query,
                "neg_prompt": neg_prompt,
                "n_images": n_images,
                "scale": scale,
                "select_seed": select_seed,
                "output_size": output_size,
                "init_img": init_img,
                "controlnet_model": controlnet_model,
            }
            r1 = requests.post(base_url + "/task_submit", json=data).json()
            task_id = r1['info']['task_id']
            logger.info(task_id)

            data2 = {
                'token': token_str,
                'task_id': task_id,
            }
            while(True):
                response2 = requests.post(base_url + "/task_result", json=data2).json()
                if response2['info']['state'] == 'done':
                    image_url = response2['info']['images'][0]['large']
                    logger.info(image_url)
                    break
                time.sleep(1)

            logger.info("[OPEN_AI] image_url={}".format(image_url))
            return True, image_url

        except openai.error.RateLimitError as e:
            logger.warn(e)
            if retry_count < 1:
                time.sleep(5)
                logger.warn("[OPEN_AI] ImgCreate RateLimit exceed, 第{}次重试".format(retry_count + 1))
                return self.create_img(query, retry_count + 1)
            else:
                return False, "提问太快啦，请休息一下再问我吧"
        except Exception as e:
            logger.exception(e)
            return False, str(e)
    
    