import React, { FC } from 'react';
import styles from './header.module.scss';

interface HeaderProps {}

const Header: FC<HeaderProps> = () => (
  <div className={styles.Header}>
    Header Component
  </div>
);

export default Header;
