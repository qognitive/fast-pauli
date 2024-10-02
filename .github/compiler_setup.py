# noqa: D100
import os
import sys

if __name__ == "__main__":
    compiler = sys.argv[1]
    os_type = sys.argv[2]

    if "macos" not in os_type:
        os.system("sudo apt-get update")

        if "g++" in compiler and "clang" not in compiler:
            version = compiler.split("-")[1]
            os.system(f"sudo apt-get -y install gcc-{version} g++-{version}")
        elif "clang" in compiler:
            version = compiler.split("-")[1]
            print(f"Installing clang {compiler}")
            os.system("wget https://apt.llvm.org/llvm.sh")
            os.system("chmod +x llvm.sh")
            os.system(f"sudo ./llvm.sh {version}")
            os.system(f"sudo apt-get install -y libomp-{version}-dev")
        else:
            raise ValueError(f"Compiler {compiler} not supported")
