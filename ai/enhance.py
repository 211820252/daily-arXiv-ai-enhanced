import os
import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
# INSERT_YOUR_CODE
import requests

import dotenv
import argparse
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from structure import Structure

if os.path.exists('.env'):
    dotenv.load_dotenv()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of parallel workers")
    return parser.parse_args()

def _extract_json(text: str) -> str:
    """从 LLM 输出中提取 JSON，容忍模型输出额外文本"""
    # 1. 尝试匹配 ```json ... ``` 代码块
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if m:
        return m.group(1)
    # 2. 尝试匹配最外层的 { ... } 对象
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        return m.group(0)
    # 3. 原样返回
    return text

def process_single_item(chain, parser, item: Dict, language: str) -> Dict:
    def is_sensitive(content: str) -> bool:
        """
        调用 spam.dw-dengwei.workers.dev 接口检测内容是否包含敏感词。
        返回 True 表示触发敏感词，False 表示未触发。
        """
        # try:
        #     resp = requests.post(
        #         "https://spam.dw-dengwei.workers.dev",
        #         json={"text": content},
        #         timeout=5
        #     )
        #     if resp.status_code == 200:
        #         result = resp.json()
        #         # 约定接口返回 {"sensitive": true/false, ...}
        #         return result.get("sensitive", True)
        #     else:
        #         # 如果接口异常，默认不触发敏感词
        #         print(f"Sensitive check failed with status {resp.status_code}", file=sys.stderr)
        #         return True
        # except Exception as e:
        #     print(f"Sensitive check error: {e}", file=sys.stderr)
        #     return True
        return False

    def check_github_code(content: str) -> Dict:
        """提取并验证 GitHub 链接"""
        code_info = {}

        # 1. 优先匹配 github.com/owner/repo 格式
        github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
        match = re.search(github_pattern, content)
        
        if match:
            owner, repo = match.groups()
            # 清理 repo 名称，去掉可能的 .git 后缀或末尾的标点
            repo = repo.rstrip(".git").rstrip(".,)")
            
            full_url = f"https://github.com/{owner}/{repo}"
            code_info["code_url"] = full_url
            
            # 尝试调用 GitHub API 获取信息
            github_token = os.environ.get("TOKEN_GITHUB")
            headers = {"Accept": "application/vnd.github.v3+json"}
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            
            try:
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                resp = requests.get(api_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    code_info["code_stars"] = data.get("stargazers_count", 0)
                    code_info["code_last_update"] = data.get("pushed_at", "")[:10]
            except Exception:
                # API 调用失败不影响主流程
                pass
            return code_info

        # 2. 如果没有 github.com，尝试匹配 github.io
        github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
        match_io = re.search(github_io_pattern, content)
        
        if match_io:
            url = match_io.group(0)
            # 清理末尾标点
            url = url.rstrip(".,)")
            code_info["code_url"] = url
            # github.io 不进行 star 和 update 判断
                
        return code_info

    # 检查 summary 字段
    if is_sensitive(item.get("summary", "")):
        return None

    # 检测代码可用性
    code_info = check_github_code(item.get("summary", ""))
    if code_info:
        item.update(code_info)

    """处理单个数据项"""
    # Default structure with meaningful fallback values
    default_ai_fields = {
        "tldr": "Summary generation failed",
        "motivation": "Motivation analysis unavailable",
        "method": "Method extraction failed",
        "result": "Result analysis unavailable",
        "conclusion": "Conclusion extraction failed"
    }
    
    try:
        response_text = chain.invoke({
            "language": language,
            "content": item['summary']
        })
        # 尝试多种方式提取 JSON（模型可能输出额外文本）
        ai_json = _extract_json(response_text)
        ai_data = parser.parse(ai_json)
        item['AI'] = ai_data.model_dump()
    except Exception as e:
        print(f"Error for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        item['AI'] = default_ai_fields
    
    # Final validation to ensure all required fields exist
    for field in default_ai_fields.keys():
        if field not in item['AI']:
            item['AI'][field] = default_ai_fields[field]

    # 检查 AI 生成的所有字段
    for v in item.get("AI", {}).values():
        if is_sensitive(str(v)):
            return None
    return item

def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int) -> List[Dict]:
    """并行处理所有数据项"""
    raw_base = os.environ.get("OPENAI_BASE_URL", "")
    # OpenAI Python 客户端会在 base_url 后拼接 /chat/completions，
    # 所以必须确保 base_url 以 /v1 结尾。DeepSeek 完整路径是 https://api.deepseek.com/v1
    if raw_base and not raw_base.rstrip('/').endswith('/v1'):
        raw_base = raw_base.rstrip('/') + '/v1'

    # 读取关键词配置文件，过滤不相关的论文（检查摘要中是否包含任意关键词）
    keywords = []
    keywords_file = os.path.join(os.path.dirname(__file__), "..", "config", "keywords.json")
    try:
        with open(keywords_file, "r") as f:
            keywords = [k.strip().lower() for k in json.load(f) if k.strip()]
    except Exception:
        pass

    skipped_ai_fields = {
        "tldr": "",
        "motivation": "",
        "method": "",
        "result": "",
        "conclusion": ""
    }

    # 分离需要 AI 处理的论文和跳过的论文
    to_process = []  # (idx, item)
    for idx, item in enumerate(data):
        if keywords:
            summary_lower = (item.get("summary", "") or "").lower()
            if not any(kw in summary_lower for kw in keywords):
                data[idx]['AI'] = skipped_ai_fields
                continue
        to_process.append((idx, item))

    print(f"Papers matching keywords: {len(to_process)} / {len(data)}", file=sys.stderr)

    if not to_process:
        return data

    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_api_base=raw_base or None,
        temperature=0.1,
    )
    print('Connect to:', model_name, 'base_url:', raw_base or 'default(OpenAI)', file=sys.stderr)

    parser = PydanticOutputParser(pydantic_object=Structure)

    system_with_format = system + "\n\n{format_instructions}"

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_with_format),
        HumanMessagePromptTemplate.from_template(template=template)
    ])
    prompt_template = prompt_template.partial(format_instructions=parser.get_format_instructions())

    chain = prompt_template | llm | StrOutputParser()

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 只提交需要处理的论文
        future_to_idx = {
            executor.submit(process_single_item, chain, parser, item, language): idx
            for idx, item in to_process
        }

        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(to_process),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                data[idx] = result
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                data[idx]['AI'] = {
                    "tldr": "Processing failed",
                    "motivation": "Processing failed",
                    "method": "Processing failed",
                    "result": "Processing failed",
                    "conclusion": "Processing failed"
                }

    return data

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print('Open:', args.data, file=sys.stderr)
    
    # 并行处理所有数据
    processed_data = process_all_items(
        data,
        model_name,
        language,
        args.max_workers
    )
    
    # 保存结果
    with open(target_file, "w") as f:
        for item in processed_data:
            if item is not None:
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
