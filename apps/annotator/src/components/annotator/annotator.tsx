import { FC } from 'react';
import CotData from '../../dtos/CotData';
import DatasetEntryElement from '../datasetentryelement/DatasetEntryElement';
import styles from './annotator.module.scss';

interface AnnotatorProps {
  username: string
  visualisationTreshold: number
  cotData: CotData[]
  anyUpdatePerformed: () => void
}

const Annotator: FC<AnnotatorProps> = (props) => {

  const entryElements = props.cotData.map(data => (<DatasetEntryElement
    key={data.id}
    cotData={data}
    anyUpdatePerformed={props.anyUpdatePerformed}
    username={props.username}
    visualisationTreshold={props.visualisationTreshold} />))

  return <div className={styles.Annotator}>
    {entryElements}
  </div>
}

export default Annotator;
