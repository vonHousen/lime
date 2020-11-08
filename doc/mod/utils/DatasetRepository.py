
import re
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetRepository:
    def __init__(self, path_to_data_dir):
        self.path_to_data_dir = path_to_data_dir

    def get_titanic_dataset(self):
        data = pd.read_csv(f"{self.path_to_data_dir}/titanic/train.csv").set_index("PassengerId")

        data.drop(columns=["Name", "Ticket"], inplace=True)
        data["cabin_level"] = data["Cabin"].str[0]
        data.drop(columns=["Cabin"], inplace=True)
        data["Age"] = data["Age"].fillna(data["Age"].mean())

        # categorical features handling
        data["Sex"] = data["Sex"].map({
            "female": 0,
            "male": 1
        })
        data = pd.concat(
            [
                data.drop(columns=["Embarked"]),
                pd.get_dummies(data["Embarked"].fillna("unknown"), prefix="embarked")
            ],
            axis=1
        )
        data = pd.concat(
            [
                data.drop(columns=["cabin_level"]),
                pd.get_dummies(data["cabin_level"].fillna("unknown"), prefix="cabin_level")
            ],
            axis=1
        )

        train_data, test_data = \
            train_test_split(data, test_size=0.1, stratify=data["Survived"])
        train_labels = train_data.pop("Survived")
        test_labels = test_data.pop("Survived")

        return Dataset("titanic", train_data, train_labels, test_data,
                       test_labels, [idx for idx in range(6, len(train_data.columns))] + [0, 1])

    def get_fetal_health_dataset(self):
        data = pd.read_csv(f"{self.path_to_data_dir}/fetal_health/fetal_health.csv")

        # data["fetal_health"] = data["fetal_health"].map({
        #     1: "Normal",
        #     2: "Suspect",
        #     3: "Pathological"
        # })

        train_data, test_data = \
            train_test_split(data, test_size=0.2, stratify=data["fetal_health"])
        train_labels = train_data.pop("fetal_health")
        test_labels = test_data.pop("fetal_health")

        return Dataset("fetal health", train_data, train_labels, test_data, test_labels, None)

    def get_wines_dataset(self):
        data = pd.read_csv(f"{self.path_to_data_dir}/wine_quality/winequality-red.csv")

        train_data, test_data = \
            train_test_split(data, test_size=0.1, stratify=data["quality"])
        train_labels = train_data.pop("quality")
        test_labels = test_data.pop("quality")

        return Dataset("wines", train_data, train_labels, test_data, test_labels, None)

    def get_mushrooms_dataset(self):
        data = pd.read_csv(f"{self.path_to_data_dir}/mushrooms/mushrooms.csv")

        legend = """
            cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
            cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
            cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
            bruises: bruises=t,no=f
            odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
            gill-attachment: attached=a,descending=d,free=f,notched=n
            gill-spacing: close=c,crowded=w,distant=d
            gill-size: broad=b,narrow=n
            gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
            stalk-shape: enlarging=e,tapering=t
            stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
            stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
            stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
            stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
            stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
            veil-type: partial=p,universal=u
            veil-color: brown=n,orange=o,white=w,yellow=y
            ring-number: none=n,one=o,two=t
            ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
            spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
            population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
            habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
        """
        dict_str = {}
        for line in legend.splitlines():
            column_match = re.findall(r"([\w-]+):", line)
            mapping_match = re.findall(r"(\w+=\w)", line)
            if len(column_match) == 1:
                column = column_match[0]
            else:
                continue
            str_representation = "{"
            for mapping in mapping_match:
                inner_mapping_match = re.findall(r"(\w+)", mapping)
                key = inner_mapping_match[1]
                value = inner_mapping_match[0]
                str_representation += f"'{key}': '{value}', "
            str_representation = str_representation[:-2] + "}"
            dict_str[column] = str_representation

        data["class"] = data["class"].map({
            "e": "edible",
            "p": "poisonous"
        })
        for column, mapping in dict_str.items():
            data[column] = data[column].map(eval(mapping))

        # categorical features handling
        data["class"] = data["class"].map({
            "edible": 0,
            "poisonous": 1
        })
        for column in data.columns.tolist()[1:]:
            data = pd.concat(
                [
                    data.drop(columns=[column]),
                    pd.get_dummies(data[column].fillna("unknown"), prefix=column)
                ],
                axis=1
            )

        train_data, test_data = \
            train_test_split(data, test_size=0.2, stratify=data["class"])
        train_labels = train_data.pop("class")
        test_labels = test_data.pop("class")

        return Dataset("mushrooms", train_data, train_labels, test_data, test_labels,
                       list(range(len(train_data.columns))))

    def get_heart_disease_dataset(self):
        data = pd.read_csv(f"{self.path_to_data_dir}/heart/heart.csv")

        train_data, test_data = \
            train_test_split(data, test_size=0.1, stratify=data["target"])
        train_labels = train_data.pop("target")
        test_labels = test_data.pop("target")

        return Dataset("heart disease", train_data, train_labels, test_data, test_labels, None)

    def get_all_datasets(self):
        dataset_getters = [
            self.get_titanic_dataset,
            self.get_fetal_health_dataset,
            self.get_wines_dataset,
            self.get_mushrooms_dataset,
            self.get_heart_disease_dataset,
        ]
        for getter in dataset_getters:
            yield getter()


class Dataset:
    def __init__(self, name, train_data, train_labels, test_data, test_labels, categorical_features):
        self.name = name
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.categorical_features = categorical_features
