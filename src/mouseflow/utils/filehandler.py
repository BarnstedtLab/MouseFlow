from pathlib import Path

def get_files2analyse(keypoint_dir, overwrite, analysis_type):
    '''
    Returns face OR body files which can be found in keypoint_dir and should be analysed.
    The files must match the pattern '*DLC*Mouse{analysis_type}*.h5'.
    '''
    if analysis_type not in ['face', 'body']:
        print("please check configuration. Only analysis_type='face' and analysis_type='body' is supported.")
    if analysis_type == 'face':
        analysis_type = 'Face'
    elif analysis_type == 'body':
        analysis_type = 'Body'
    if keypoint_dir is None:
        raise RuntimeError("No DLC directory defined.")
    d = Path(keypoint_dir)

    all_files = sorted(d.glob(f'*DLC*Mouse{analysis_type}*.h5'))

    def needs_processing(p: Path) -> bool:
        if p.name.lower().endswith("_mouseflow.h5"):
            return False
        mf = p.with_name(p.stem + '_mouseflow.h5')
        if mf.exists() and not overwrite:
            print(f"skipping file {p.name} as it is already analysed!") 
        return overwrite or not mf.exists()

    rtrn = [p for p in all_files if needs_processing(p)]
    return rtrn