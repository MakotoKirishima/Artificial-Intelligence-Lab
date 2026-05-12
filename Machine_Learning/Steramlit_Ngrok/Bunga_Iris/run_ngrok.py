import sys
import time
import subprocess
from pyngrok import ngrok


PORT = 8501


def main():
    authtoken = input("Enter your ngrok authtoken: ")

    ngrok.set_auth_token(authtoken)
    ngrok.kill()

    streamlit_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "apps.py",
            "--server.port",
            str(PORT),
            "--server.headless",
            "true"
        ]
    )

    time.sleep(5)

    public_url = ngrok.connect(PORT)

    print("\nStreamlit app URL:")
    print(public_url.public_url)
    print("\nPress CTRL + C to stop.\n")

    try:
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nStopping app...")
        ngrok.kill()
        streamlit_process.terminate()


if __name__ == "__main__":
    main()