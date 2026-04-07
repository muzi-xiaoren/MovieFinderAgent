"""程序入口"""

import sys
from movieFinder import MovieFinder


def main():
    print("初始化 MovieFinder Agent...\n")
    finder = MovieFinder()
    
    print("输入问题后按回车，输入 exit / quit / 退出 结束\n")
    
    while True:
        try:
            user_input = input(":").strip()
            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "退出", "bye"}:
                print("\n再见！")
                break

            # 只调用 chat，让 chat 内部负责流式输出
            finder.chat(user_input)
            print("-" * 60)   # 分隔线放在这里

        except KeyboardInterrupt:
            print("\n\n已退出")
            sys.exit(0)
        except Exception as e:
            print(f"\n错误: {e}")
            print("-" * 60)


if __name__ == "__main__":
    main()