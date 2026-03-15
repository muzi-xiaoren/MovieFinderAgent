"""程序入口"""

import sys
from movieFinder import MovieFinder


def main():
    print("初始化 MovieFinder Agent...\n")
    finder = MovieFinder()
    
    print("输入问题后按回车，输入 exit / quit / 退出 结束\n")
    
    while True:
        try:
            user_input = input("你: ").strip()
            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "退出", "bye"}:
                print("\n再见！")
                break

            # Agent 处理
            response = finder.chat(user_input)
            
            print("\nAI: ", end="", flush=True)
            print(response)
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\n已退出")
            sys.exit(0)
        except Exception as e:
            print(f"\n错误: {e}")
            print("-" * 60)


if __name__ == "__main__":
    main()