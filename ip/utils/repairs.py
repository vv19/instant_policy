import regex
import torch


def remove_prefix(text, prefix):
    if prefix in text:
        return regex.sub(prefix, "", text)
    # if text.startswith(prefix):
    #     return text[len(prefix) :]
    return text


def repair_checkpoint(path, save_path=None):
    ckpt = torch.load(path)
    in_state_dict = ckpt["state_dict"]
    pairings = [
        (src_key, remove_prefix(src_key, "_orig_mod."))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        return  # Do not write checkpoint if no need to repair!
    out_state_dict = {}
    for src_key, dest_key in pairings:
        out_state_dict[dest_key] = in_state_dict[src_key]
    ckpt["state_dict"] = out_state_dict
    torch.save(ckpt, save_path)
