import { FC, useState } from 'react';
import styles from './dropdown.module.scss';

interface DropdownProps {
  currentChoice?: string
  options: string[]
  onClick: (option: string) => void
}

const Dropdown: FC<DropdownProps> = (props) => {
  const [droppedDown, setDroppedDown] = useState(false)

  const optionElements = props.options.map(option => (
    <li onClick={() => {
      setDroppedDown(false)
      props.onClick(option)
    }}
      className={props.currentChoice == option ? styles.selected : ""}>
      {option}
    </li>))

  return <div className={styles.Dropdown}>
    <div onClick={() => setDroppedDown(!droppedDown)}>
      <span>{props.currentChoice} </span>
      <i className="fa-solid fa-caret-down"></i>
    </div>
    {
      droppedDown ?
        <ul>{optionElements}</ul>
        :
        null
    }
  </div>

}

export default Dropdown;
