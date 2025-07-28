from rag import Rag


def main():
    agent = Rag()
    agent.load_manifest("data/caption_manifest.json")
    response = agent.ask(
        q="What areas show signs of poor infrastructure?",
        k=3
    )
    
    print(response)


if __name__ == "__main__":
    main()
