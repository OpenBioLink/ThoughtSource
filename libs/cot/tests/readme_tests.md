If you want to use tests make sure to define the working directory for pytest to "./libs/cot".

For working in VScode add this to your settings.json:
```
{
    "python.testing.cwd": "./libs/cot",
    "python.testing.pytestArgs": [
        "tests",
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true
}
```