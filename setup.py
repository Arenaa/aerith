import os
import subprocess
import sys

def run_command(command):
    """Execute a shell command and print output."""
    try:
        process = subprocess.run(command, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
        print(process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

def setup_project():
    """Set up the project environment."""
    # Clone T2IScoreScore repository
    if not os.path.exists('T2IScoreScore'):
        print("Cloning T2IScoreScore repository...")
        run_command("git clone https://github.com/michaelsaxon/T2IScoreScore.git")
    
    # Install requirements
    print("Installing requirements...")
    run_command("pip install -r requirements.txt")
    
    # Install T2IScoreScore
    print("Installing T2IScoreScore...")
    os.chdir('T2IScoreScore')
    run_command("pip install -e .")
    os.chdir('..')

if __name__ == "__main__":
    setup_project()
    print("Setup completed successfully!") 