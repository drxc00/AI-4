from rag import Rag

def main():
    agent = Rag()
    agent.load_manifest("data/caption_manifest.json")
    agent.run()


if __name__ == "__main__":
    main()
