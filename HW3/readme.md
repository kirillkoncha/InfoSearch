Корпус должен называться 'questions_about_love.jsonl' и лежать в одной папке с программой. 

В этот раз было решено не удалять стоп-слова при препроцессинге, так как они на самом деле играют роль при поиске. Например, допустим есть документы 'бить его', 'бить баклуши' и 'бить все'. Если мы введем 'бить все', то без стоп слов максимальный скор будет у 'бить все', что нам и надо. НО, если удалить стоп-слова, то 'бить все' и 'бить его' превратятся просто в 'бить', что confusing.