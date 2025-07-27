import ollama

def main():
    res = ollama.chat(
        model="moondream:1.8b",
        messages=[
            {
                "role":"user",
                "content": "Describe this image and generate tags. Response in JSON format with 'caption' and 'tags' keys.",
                "images": ["./images/pasig-river-dredging.jpg"]
            }
        ]
    )
    
    print(res['message']['content'])

if __name__ == "__main__":
    main()