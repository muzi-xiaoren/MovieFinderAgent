# @tool 装饰器 ≠ MCP
# 但 用 @tool 写出来的工具 + langchain-mcp-adapters 可以变成 MCP 工具
# 反过来，MCP server 的工具也能被适配器转成 LangChain 的 Tool 对象，双方现在是可以互相桥接的。、
# LangChain 的 @tool 是“写工具最方便的方式”，MCP 是“把工具做成标准 USB 接口”的方式。

"""豆瓣电影搜索工具"""

import re
import time
from urllib.parse import quote
from typing import List, Dict
from playwright.sync_api import sync_playwright
from langchain.tools import tool

# 手动实现 stealth 脚本注入（不依赖 playwright_stealth 包）
STEALTH_SCRIPTS = """
() => {
    // 隐藏 webdriver 标记
    Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined
    });
    
    // 伪装 plugins
    Object.defineProperty(navigator, 'plugins', {
        get: () => [1, 2, 3, 4, 5]
    });
    
    // 伪装 languages
    Object.defineProperty(navigator, 'languages', {
        get: () => ['zh-CN', 'zh', 'en']
    });
    
    // 覆盖 chrome 属性
    window.chrome = {
        runtime: {}
    };
    
    // 删除 automation 标记
    delete navigator.__proto__.webdriver;
}
"""

"""豆瓣电影搜索工具"""

def extract_movie_links(html_content: str, max_results: int = 5) -> List[str]:
    """从搜索结果页提取电影链接"""
    pattern = r'href="(https://movie\.douban\.com/subject/\d+/)"'
    matches = re.findall(pattern, html_content)
    
    seen = set()
    unique_links = []
    for link in matches:
        if link not in seen and "subject" in link:
            seen.add(link)
            unique_links.append(link)
            if len(unique_links) >= max_results:
                break
    
    return unique_links


def extract_movie_detail(page, url: str) -> Dict:
    """访问详情页提取信息"""
    # print(f"  访问详情页: {url}")
    
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
        time.sleep(2)  # 等待渲染
        
        # 提取 info 区域
        info_html = ""
        info_elem = page.locator('#info').first
        if info_elem.count() > 0:
            info_html = info_elem.inner_html()
            # 清理 HTML 标签，保留文本
            info_text = re.sub(r'<[^>]+>', '', info_html)
            info_text = re.sub(r'\s+', ' ', info_text).strip()
        else:
            info_text = "未找到信息"
        
        # 提取简介 v:summary
        summary = ""
        summary_elem = page.locator('[property="v:summary"]').first
        if summary_elem.count() > 0:
            summary = summary_elem.inner_text()
            summary = re.sub(r'\s+', ' ', summary).strip()
        else:
            # 尝试其他选择器
            summary_elem = page.locator('.related-info .indent span').first
            if summary_elem.count() > 0:
                summary = summary_elem.inner_text().strip()
        
        return {
            "url": url,
            "success": True,
            "info": info_text,
            "summary": summary
        }
        
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": str(e)
        }


@tool
def search_douban_movies(movie_name: str) -> str:
    """
    搜索豆瓣电影信息，返回详细信息。
    
    Args:
        movie_name: 电影名称，如"肖申克的救赎"
    
    Returns:
        返回前3搜索结果的详细信息（info和简介）
    """
    encoded_name = quote(movie_name)
    search_url = f"https://search.douban.com/movie/subject_search?search_text={encoded_name}&cat=1002"
    
    # print(f"正在搜索: {movie_name}")
    # print(f"URL: {search_url}")
    
    results = []
    
    with sync_playwright() as p:
        # 启动浏览器
        browser = p.chromium.launch(
            headless=True,
            args=['--disable-blink-features=AutomationControlled']
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="zh-CN",
        )
        page = context.new_page()
        page.add_init_script(STEALTH_SCRIPTS)
        
        try:
            # 第一步：搜索
            page.goto(search_url, wait_until="networkidle", timeout=30000)
            time.sleep(3)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(2)
            
            html_content = page.content()
            movie_links = extract_movie_links(html_content, max_results=3)
            
            # print(f"找到 {len(movie_links)} 个电影:")
            # for i, link in enumerate(movie_links, 1):
            #     print(f"  [{i}] {link}")
            
            if not movie_links:
                browser.close()
                return "未找到相关电影"
            
            # 第二步：依次访问详情页
            for link in movie_links:
                detail = extract_movie_detail(page, link)
                if detail["success"]:
                    results.append(
                        f"【电影信息】\n"
                        f"链接: {detail['url']}\n"
                        f"基本信息:\n{detail['info']}\n\n"
                        f"简介:\n{detail['summary'][:500]}..."
                    )
                else:
                    results.append(f"查询失败 {link}: {detail['error']}")
                time.sleep(1)  # 防反爬
            
            browser.close()
            
        except Exception as e:
            browser.close()
            return f"搜索失败: {str(e)}"
    
    return "\n\n" + "="*50 + "\n\n".join(results)


if __name__ == "__main__":
    result = search_douban_movies.invoke({"movie_name": "肖申克的救赎"})
    print("\n最终结果:")
    print(result)