# ThoughtSource
This repository is dedicated to datasets that are converted to our intended COT format.

## Dataloader Usage

1. Clone repository
2. Run `pip install -e ./dataloader`
   
```python
from dataloader import load_datasets
a = load_datasets(["gsm8k", "open_book_qa"])
print(a)
print(a["gsm8k"]["train"][0])
```

<h2>Datasets</h2>
<p>The following table represents statistics of datasets we converted to the COT format.</p>
</p>
<table>
  <tr>
    <th>
      Dataset
    </th>
    <th width="20">
      Training samples
    </th>
    <th>
      Dev samples
    </th>
    <th>
      Test samples
    </th>
  </tr>
  <tr>
    <td>
      GSM8K
    </td>
    <td>
      7473
    </td>
    <td>
      -
    </td>
    <td>
      1319
    </td>
  </tr>
  <tr>
    <td>
      StrategyQA
    </td>
    <td>
      2290
    </td>
    <td>
      -
    </td>
    <td>
      490
    </td>
  </tr>
  <tr>
    <td>
      QED
    </td>
    <td>
      5154
    </td>
    <td>
      -
    </td>
    <td>
      1021
    </td>
  </tr>
  <tr>
    <td>
      OpenBookQA
    </td>
    <td>
      4957
    </td>
    <td>
      500
    </td>
    <td>
      500
    </td>
  </tr>
  <tr>
    <td>
      WorldTree
    </td>
    <td>
      2206
    </td>
    <td>
      496
    </td>
    <td>
      1663
    </td>
  </tr>
</table>
