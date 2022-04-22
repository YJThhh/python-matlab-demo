# Evaluation Toolbox

## Dependencies

- Python3
- Python packages in `requirements.txt`
- MATLAB Engine API for Python

## Installation

### A. Install MATLAB Engine API for Python

#### a. Windows

##### 1. Find your MATLAB root

```bash
where matlab
```

For example:

```bash
C:\Program Files\MATLAB\R2019b\bin\matlab.exe
```

##### 2. Go to the Python API folder

```bash
cd matlabroot\extern\engines\python
```

For example:

```bash
cd C:\Program Files\MATLAB\R2019b\extern\engines\python
```

##### 3. Install MATLAB Engine API for Python

```bash
python setup.py install
```

#### b. Ubuntu

##### 1. Find your MATLAB root

```bash
sudo find / -name MATLAB
```

For example:

```bash
# default MATLAB root
/usr/local/MATLAB/R2016b
```

##### 2. Go to Python API folder

```bash
cd matlabroot/extern/engines/python
```

For example:

```bash
cd /usr/local/MATLAB/R2016b/extern/engines/python
```

##### 3. Install MATLAB Engine API for Python

```bash
python setup.py install
```

### B. Install Required Modules

```bash
pip install -r requirements.txt
```
