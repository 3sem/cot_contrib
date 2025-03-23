import compiler_gym
from compiler_gym.envs import LlvmEnv
import time
import subprocess

# Create the environment
def make_bc_dump(bench_name="bzip2"):
    env = compiler_gym.make("llvm-v0")

    # Add a custom pass to the optimization pipeline
    env.reset(benchmark="cBench-v1/" + bench_name)  # Load a benchmark from cBench
    env_obs_code_size = env.observation["TextSizeOz"]
    # Run the environment
    observation, reward, done, info = env.step(env.action_space.sample())
    env.write_bitcode(bench_name + ".bc")
    env.close()
    return bench_name + ".bc", env_obs_code_size

def compile_bc_to_executable(bc_file_path, output_file="a.out", flagsstr="-Oz"):
    """
    Compile a .bc file to an executable using clang.

    :param bc_file_path: Path to the input .bc file.
    :param output_file: Name of the output executable (default is "a.out").
    """
    try:
        # Command to compile .bc to executable with flagsstr
        command = ["clang", flagsstr, bc_file_path, "-o", output_file]

        # Run the command
        result = subprocess.run(command, check=True,
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)

        # Print success message
        # print(f"Successfully compiled {bc_file_path} to {output_file}.")

    except subprocess.CalledProcessError as e:
        # Handle errors
        print(f"Error occurred while compiling {bc_file_path}:")
        print(e.stderr.decode())
        
    return output_file
    

def validator(
    benchmark: str,
    cmd: str,
    data = None,
    outs = None,
    env = None,
    linkopts = None
    
):
    """
    A simplified validator function that returns a dictionary with keys:
    - benchmark: The name of the benchmark.
    - cmd: The command to run the benchmark.
    - data: A list of input file paths.
    """
    return {
        "benchmark": benchmark,
        "cmd": cmd,
        "data": data or [],  # Ensure data is always a list, even if None
    }

def populate_tests(BIN="a.out", DIR_PREFIX="cbench") -> dict:
    i=1
    res = {}
    ret = validator(benchmark="bitcount", cmd=f"{BIN} 1125000")
    res[ret['benchmark']] = ret
    ret = validator(benchmark="blowfish", cmd=f"{BIN} d {DIR_PREFIX}/office_data/{i}.benc output.txt 1234567890abcdeffedcba0987654321", data=[f"office_data/{i}.benc"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="bzip2", cmd=f"{BIN} -d -k -f -c {DIR_PREFIX}/bzip2_data/{i}.bz2", data=[f"bzip2_data/{i}.bz2"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="crc32", cmd=f"{BIN} {DIR_PREFIX}/telecom_data/{i}.pcm", data=[f"telecom_data/{i}.pcm"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="dijkstra", cmd=f"{BIN} {DIR_PREFIX}/network_dijkstra_data/{i}.dat", data=[f"network_dijkstra_data/{i}.dat"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="gsm", cmd=f"{BIN} -fps -c {DIR_PREFIX}/telecom_gsm_data/{i}.au", data=[f"telecom_gsm_data/{i}.au"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="jpeg-c", cmd=f"{BIN} -dct int -progressive -outfile output.jpeg {DIR_PREFIX}/consumer_jpeg_data/{i}.ppm", data=[f"consumer_jpeg_data/{i}.ppm"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="jpeg-d", cmd=f"{BIN} -dct int -outfile output.ppm {DIR_PREFIX}/consumer_jpeg_data/{i}.jpg", data=[f"consumer_jpeg_data/{i}.jpg"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="patricia", cmd=f"{BIN} {DIR_PREFIX}/network_patricia_data/{i}.udp", data=[f"network_patricia_data/{i}.udp"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="qsort", cmd=f"{BIN} {DIR_PREFIX}/automotive_qsort_data/{i}.dat", data=[f"automotive_qsort_data/{i}.dat"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="sha", cmd=f"{BIN} {DIR_PREFIX}/office_data/{i}.txt", data=[f"office_data/{i}.txt"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="stringsearch", cmd=f"{BIN} {DIR_PREFIX}/office_data/{i}.txt {DIR_PREFIX}/office_data/{i}.s.txt output.txt", data=[f"office_data/{i}.txt"], outs=["output.txt"], env={}, linkopts=["-lm"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="stringsearch2", cmd=f"{BIN} {DIR_PREFIX}/office_data/{i}.txt {DIR_PREFIX}/office_data/{i}.s.txt output.txt", data=[f"office_data/{i}.txt"], outs=["output.txt"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="susan", cmd=f"{BIN} {DIR_PREFIX}/automotive_susan_data/{i}.pgm output_large.corners.pgm -c", data=[f"automotive_susan_data/{i}.pgm"], outs=["output_large.corners.pgm"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="tiff2bw", cmd=f"{BIN} {DIR_PREFIX}/consumer_tiff_data/{i}.tif output.tif", data=[f"consumer_tiff_data/{i}.tif"], outs=["output.tif"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="tiff2rgba", cmd=f"{BIN} {DIR_PREFIX}/consumer_tiff_data/{i}.tif output.tif", data=[f"consumer_tiff_data/{i}.tif"], outs=["output.tif"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="tiffdither", cmd=f"{BIN} {DIR_PREFIX}/consumer_tiff_data/{i}.bw.tif out.tif", data=[f"consumer_tiff_data/{i}.bw.tif"], outs=["out.tif"])
    res[ret['benchmark']] = ret
    ret = validator(benchmark="tiffmedian", cmd=f"{BIN} {DIR_PREFIX}/consumer_tiff_data/{i}.nocomp.tif output.tif", data=[f"consumer_tiff_data/{i}.nocomp.tif"], outs=["output.tif"])
    res[ret['benchmark']] = ret
    return res

def prepare_run_data() -> dict:
    return populate_tests()


def evaluate(benchmark="crc32"):
    pt = prepare_run_data()
    start_time = time.perf_counter()

    ret = subprocess.run(pt[benchmark]['cmd'], shell=True, text=True, capture_output=True)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return ret, elapsed_time


def prepare_baselines(benchmark):
    name, size = make_bc_dump(bench_name=benchmark)
    out_f = compile_bc_to_executable(name)
    res, runtime = evaluate(benchmark=benchmark)
    return runtime, size
