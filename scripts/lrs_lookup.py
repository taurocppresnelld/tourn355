import json 
import os 
import hashlib
current_dir = os.path.dirname(os.path.abspath(__file__))



with open(os.path.join(current_dir, "ratio/dpo.json"), "r") as f:
    dpo_ratio = json.load(f)

with open(os.path.join(current_dir, "ratio/instruct.json"), "r") as f:
    instruct_ratio = json.load(f)

with open(os.path.join(current_dir, "ratio/grpo.json"), "r") as f:
    grpo_ratio = json.load(f)

def find_lr_ratio_dpo(model: str):
    hashed_model = hash_model(model)
    for ratio in dpo_ratio:
        if ratio["h"] == hashed_model:
            return ratio["ratio"]
    return 1.0

def find_lr_ratio_instruct(model: str):
    hashed_model = hash_model(model)
    for ratio in instruct_ratio:
        if ratio["h"] == hashed_model:
            return ratio["ratio"]
    return 1.0

def find_lr_ratio_grpo(model: str):
    hashed_model = hash_model(model)
    for ratio in grpo_ratio:
        if ratio["h"] == hashed_model:
            return ratio["ratio"]
    return 1.0


def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 


def get_dpo_lr(model: str):
    scale_factor = 1.0
    hashed_model = hash_model(model)
    print(f"model_name: {model}", flush=True)

    config_file = f"{os.path.join(current_dir, 'lrs')}/archives/dpo_{model.split('/', 1)[1]}.json"
    print(f"config_dpo1: {config_file}")
    if os.path.exists(config_file):
        print(f"Config: {config_file}")
        with open(config_file, "r") as f:
            dpo_lrs = json.load(f)
    else:
        config_file = f"{os.path.join(current_dir, 'lrs')}/archives/dpo_{model.split('/', 1)[0]}.json"
        print(f"config_dpo0: {config_file}")
        if os.path.exists(config_file):
            print(f"Config: {config_file}")
            with open(config_file, "r") as f:
                dpo_lrs = json.load(f)
        else:
            config_file = f"{os.path.join(current_dir, 'lrs')}/dpo.json"
            print(f"config_dpo_default: {config_file}")
            if os.path.exists(config_file):
                print(f"Config: {config_file}")
                with open(config_file, "r") as f:
                    dpo_lrs = json.load(f)

    ratio_file = f"{os.path.join(current_dir, 'lrs')}/archives/ratio_dpo_{model.split('/', 1)[1]}.json"
    print(f"ratio_dpo1: {ratio_file}")
    if os.path.exists(ratio_file):
        print(f"Ratio: {ratio_file}")
        with open(ratio_file, "r") as f:
            ratio_lrs = json.load(f)
            scale_factor = ratio_lrs["ratio"]
    else:
        ratio_file = f"{os.path.join(current_dir, 'lrs')}/archives/ratio_dpo_{model.split('/', 1)[0]}.json"
        print(f"ratio_dpo0: {ratio_file}")
        if os.path.exists(ratio_file):
            print(f"Ratio: {ratio_file}")
            with open(ratio_file, "r") as f:
                ratio_lrs = json.load(f)
                scale_factor = ratio_lrs["ratio"]
        else:
            ratio = find_lr_ratio_dpo(model)

            if ratio == 1.0:
                ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_dpo.json"
                print(f"ratio_dpo_default: {ratio_file}")
                if os.path.exists(ratio_file):
                    print(f"Ratio: {ratio_file}")
                    with open(ratio_file, "r") as f:
                        ratio_lrs = json.load(f)
                        scale_factor = ratio_lrs["ratio"]
            else:
                print(f"ratio_dpo_default_hash: ")
                scale_factor = ratio

    for lr in dpo_lrs:
        if lr["h"] == hashed_model:
            lr_return = lr["lr"] * scale_factor
            print(f"scale: {scale_factor}")
            print(f"lr: {lr_return}")
            return lr_return

    return None


def get_grpo_lr(model: str):
    scale_factor = 1.0
    hashed_model = hash_model(model)
    print(f"model_name: {model}", flush=True)

    config_file = f"{os.path.join(current_dir, 'lrs')}/archives/grpo_{model.split('/', 1)[1]}.json"
    print(f"config_grpo1: {config_file}")
    if os.path.exists(config_file):
        print(f"Config: {config_file}")
        with open(config_file, "r") as f:
            grpo_lrs = json.load(f)
    else:
        config_file = f"{os.path.join(current_dir, 'lrs')}/archives/grpo_{model.split('/', 1)[0]}.json"
        print(f"config_grpo0: {config_file}")
        if os.path.exists(config_file):
            print(f"Config: {config_file}")
            with open(config_file, "r") as f:
                grpo_lrs = json.load(f)
        else:
            config_file = f"{os.path.join(current_dir, 'lrs')}/grpo.json"
            print(f"config_grpo_default: {config_file}")
            if os.path.exists(config_file):
                print(f"Config: {config_file}")
                with open(config_file, "r") as f:
                    grpo_lrs = json.load(f)

    ratio_file = f"{os.path.join(current_dir, 'lrs')}/archives/ratio_grpo_{model.split('/', 1)[1]}.json"
    print(f"ratio_grpo1: {ratio_file}")
    if os.path.exists(ratio_file):
        print(f"Ratio: {ratio_file}")
        with open(ratio_file, "r") as f:
            ratio_lrs = json.load(f)
            scale_factor = ratio_lrs["ratio"]
    else:
        ratio_file = f"{os.path.join(current_dir, 'lrs')}/archives/ratio_grpo_{model.split('/', 1)[0]}.json"
        print(f"ratio_grpo0: {ratio_file}")
        if os.path.exists(ratio_file):
            print(f"Ratio: {ratio_file}")
            with open(ratio_file, "r") as f:
                ratio_lrs = json.load(f)
                scale_factor = ratio_lrs["ratio"]
        else:
            ratio = find_lr_ratio_grpo(model)

            if ratio == 1.0:
                ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_grpo.json"
                print(f"ratio_grpo_default: {ratio_file}")
                if os.path.exists(ratio_file):
                    print(f"Ratio: {ratio_file}")
                    with open(ratio_file, "r") as f:
                        ratio_lrs = json.load(f)
                        scale_factor = ratio_lrs["ratio"]
            else:
                print(f"ratio_grpo_default_hash: ")
                scale_factor = ratio

    for lr in grpo_lrs:
        if lr["h"] == hashed_model:
            lr_return = lr["lr"] * scale_factor
            print(f"scale: {scale_factor}")
            print(f"lr: {lr_return}")
            return lr_return

    return None


def get_grpo_lr_ratio(model: str):
    scale_factor = 1.0
    hashed_model = hash_model(model)
    print(f"model_name: {model}", flush=True)

    config_file = f"{os.path.join(current_dir, 'lrs')}/archives/grpo_{model.split('/', 1)[1]}.json"
    print(f"config_grpo1: {config_file}")
    if os.path.exists(config_file):
        print(f"Config: {config_file}")
        with open(config_file, "r") as f:
            grpo_lrs = json.load(f)
    else:
        config_file = f"{os.path.join(current_dir, 'lrs')}/archives/grpo_{model.split('/', 1)[0]}.json"
        print(f"config_grpo0: {config_file}")
        if os.path.exists(config_file):
            print(f"Config: {config_file}")
            with open(config_file, "r") as f:
                grpo_lrs = json.load(f)
        else:
            config_file = f"{os.path.join(current_dir, 'lrs')}/grpo.json"
            print(f"config_grpo_default: {config_file}")
            if os.path.exists(config_file):
                print(f"Config: {config_file}")
                with open(config_file, "r") as f:
                    grpo_lrs = json.load(f)

    ratio_file = f"{os.path.join(current_dir, 'lrs')}/archives/ratio_grpo_{model.split('/', 1)[1]}.json"
    print(f"ratio_grpo1: {ratio_file}")
    if os.path.exists(ratio_file):
        print(f"Ratio: {ratio_file}")
        with open(ratio_file, "r") as f:
            ratio_lrs = json.load(f)
            scale_factor = ratio_lrs["ratio"]
    else:
        ratio_file = f"{os.path.join(current_dir, 'lrs')}/archives/ratio_grpo_{model.split('/', 1)[0]}.json"
        print(f"ratio_grpo0: {ratio_file}")
        if os.path.exists(ratio_file):
            print(f"Ratio: {ratio_file}")
            with open(ratio_file, "r") as f:
                ratio_lrs = json.load(f)
                scale_factor = ratio_lrs["ratio"]
        else:
            ratio = find_lr_ratio_grpo(model)

            if ratio == 1.0:
                ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_grpo.json"
                print(f"ratio_grpo_default: {ratio_file}")
                if os.path.exists(ratio_file):
                    print(f"Ratio: {ratio_file}")
                    with open(ratio_file, "r") as f:
                        ratio_lrs = json.load(f)
                        scale_factor = ratio_lrs["ratio"]
            else:
                print(f"ratio_grpo_default_hash: ")
                scale_factor = ratio

    for lr in grpo_lrs:
        if lr["h"] == hashed_model:
            lr_return = lr["lr"] * scale_factor
            print(f"scale: {scale_factor}")
            print(f"lr: {lr_return}")
            return scale_factor

    return 1.0


def get_instruct_lr(model: str):
    scale_factor = 1.0
    hashed_model = hash_model(model)
    print(f"model_name: {model}", flush=True)

    config_file = f"{os.path.join(current_dir, 'lrs')}/archives/instruct_{model.split('/', 1)[1]}.json"
    print(f"config_instruct1: {config_file}")
    if os.path.exists(config_file):
        print(f"Config: {config_file}")
        with open(config_file, "r") as f:
            instruct_lrs = json.load(f)
    else:
        config_file = f"{os.path.join(current_dir, 'lrs')}/archives/instruct_{model.split('/', 1)[0]}.json"
        print(f"config_instruct0: {config_file}")
        if os.path.exists(config_file):
            print(f"Config: {config_file}")
            with open(config_file, "r") as f:
                instruct_lrs = json.load(f)
        else:
            config_file = f"{os.path.join(current_dir, 'lrs')}/instruct.json"
            print(f"config_instruct_default: {config_file}")
            if os.path.exists(config_file):
                print(f"Config: {config_file}")
                with open(config_file, "r") as f:
                    instruct_lrs = json.load(f)

    ratio_file = f"{os.path.join(current_dir, 'lrs')}/archives/ratio_instruct_{model.split('/', 1)[1]}.json"
    print(f"ratio_instruct1: {ratio_file}")
    if os.path.exists(ratio_file):
        print(f"Ratio: {ratio_file}")
        with open(ratio_file, "r") as f:
            ratio_lrs = json.load(f)
            scale_factor = ratio_lrs["ratio"]
    else:
        ratio_file = f"{os.path.join(current_dir, 'lrs')}/archives/ratio_instruct_{model.split('/', 1)[0]}.json"
        print(f"ratio_instruct0: {ratio_file}")
        if os.path.exists(ratio_file):
            print(f"Ratio: {ratio_file}")
            with open(ratio_file, "r") as f:
                ratio_lrs = json.load(f)
                scale_factor = ratio_lrs["ratio"]
        else:
            ratio = find_lr_ratio_instruct(model)

            if ratio == 1.0:
                ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_instruct.json"
                print(f"ratio_instruct_default: {ratio_file}")
                if os.path.exists(ratio_file):
                    print(f"Ratio: {ratio_file}")
                    with open(ratio_file, "r") as f:
                        ratio_lrs = json.load(f)
                        scale_factor = ratio_lrs["ratio"]
            else:
                print(f"ratio_instruct_default_hash: ")
                scale_factor = ratio


    for lr in instruct_lrs:
        if lr["h"] == hashed_model:
            lr_return = lr["lr"] * scale_factor
            print(f"scale: {scale_factor}")
            print(f"lr: {lr_return}")
            return lr_return

    return None


def get_grpo_python_lr(model: str):
    scale_factor = 1.0
    hashed_model = hash_model(model)
    print(f"model_name: {model}", flush=True)

    config_file = f"{os.path.join(current_dir, 'lrs')}/archives/grpo_python_{model.split('/', 1)[1]}.json"
    print(f"config_grpo_python1: {config_file}")
    if os.path.exists(config_file):
        print(f"Config: {config_file}")
        with open(config_file, "r") as f:
            grpo_python_lrs = json.load(f)
    else:
        config_file = f"{os.path.join(current_dir, 'lrs')}/archives/grpo_python_{model.split('/', 1)[0]}.json"
        print(f"config_grpo_python0: {config_file}")
        if os.path.exists(config_file):
            print(f"Config: {config_file}")
            with open(config_file, "r") as f:
                grpo_python_lrs = json.load(f)
        else:
            config_file = f"{os.path.join(current_dir, 'lrs')}/grpo_python.json"
            print(f"config_grpo_python_default: {config_file}")
            if os.path.exists(config_file):
                print(f"Config: {config_file}")
                with open(config_file, "r") as f:
                    grpo_python_lrs = json.load(f)

    ratio_file = f"{os.path.join(current_dir, 'lrs')}/archives/ratio_grpo_python_{model.split('/', 1)[1]}.json"
    print(f"ratio_grpo_python1: {ratio_file}")
    if os.path.exists(ratio_file):
        print(f"Ratio: {ratio_file}")
        with open(ratio_file, "r") as f:
            ratio_lrs = json.load(f)
            scale_factor = ratio_lrs["ratio"]
    else:
        ratio_file = f"{os.path.join(current_dir, 'lrs')}/archives/ratio_grpo_python_{model.split('/', 1)[0]}.json"
        print(f"ratio_grpo_python0: {ratio_file}")
        if os.path.exists(ratio_file):
            print(f"Ratio: {ratio_file}")
            with open(ratio_file, "r") as f:
                ratio_lrs = json.load(f)
                scale_factor = ratio_lrs["ratio"]
        else:
            ratio = find_lr_ratio_grpo(model)

            if ratio == 1.0:
                ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_grpo_python.json"
                print(f"ratio_grpo_python_default: {ratio_file}")
                if os.path.exists(ratio_file):
                    print(f"Ratio: {ratio_file}")
                    with open(ratio_file, "r") as f:
                        ratio_lrs = json.load(f)
                        scale_factor = ratio_lrs["ratio"]
            else:
                print(f"ratio_grpo_python_default_hash: ")
                scale_factor = ratio

    for lr in grpo_python_lrs:
        if lr["h"] == hashed_model:
            lr_return = lr["lr"] * scale_factor
            print(f"scale: {scale_factor}")
            print(f"lr: {lr_return}")
            return lr_return

    return None
