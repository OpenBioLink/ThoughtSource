import { FC } from 'react';
import styles from './dropdown.module.scss';

interface DropdownProps {
  options: Record<string, ((key: string) => void)>
}

const Dropdown: FC<DropdownProps> = () => (
  <div className={styles.Dropdown}>
    Dropdown Component
  </div>
);

export default Dropdown;
