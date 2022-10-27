import { FC } from 'react';
import styles from './dropdown.module.scss';

interface DropdownProps {
  options: string[]
  onClick: (option: string) => void
}

const Dropdown: FC<DropdownProps> = (props) => {

  const optionElements = props.options.map(option => (
    <li onClick={() => props.onClick(option)}>
      {option}
    </li>))

  return <ul className={styles.Dropdown}>
    {optionElements}
  </ul>
}

export default Dropdown;
